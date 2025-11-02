import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import docx
import io
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import base64
from datetime import datetime

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    st.warning("Downloading NLTK data... This may take a moment.")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Page configuration
st.set_page_config(
    page_title="ResumePolish - AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        padding: 20px;
    }
    .sub-header {
        font-size: 2rem;
        color: #2E86AB;
        margin-bottom: 1rem;
        font-weight: bold;
        border-left: 5px solid #2E86AB;
        padding-left: 15px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 0.5rem;
        color: white;
        text-align: center;
    }
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 15px 0;
        border: 2px solid #e0e0e0;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    .positive-feedback {
        color: #27ae60;
        font-weight: bold;
        background: #d5f4e6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
    }
    .improvement-feedback {
        color: #e74c3c;
        font-weight: bold;
        background: #fadbd8;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    }
    .upload-area {
        border: 3px dashed #2E86AB;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background: #f8f9fa;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Initialize sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

# Main title with gradient effect
st.markdown('<div class="main-header">📄 ResumePolish - AI Resume Analyzer</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🚀 Navigation")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Choose a section", 
    ["🏠 Home", "📤 Upload Resume", "📊 Analysis Results", "💡 Tips & Suggestions"])

# Function to extract text from different file types
def extract_text_from_file(uploaded_file):
    text = ""
    
    try:
        if uploaded_file.type == "application/pdf":
            # For PDF files using pdfplumber (more reliable)
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # For DOCX files
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
        
        return text.strip()
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return ""

# Function to analyze resume
def analyze_resume(text):
    analysis = {}
    
    # Basic statistics
    words = text.split()
    analysis['word_count'] = len(words)
    analysis['char_count'] = len(text)
    sentences = re.split(r'[.!?]+', text)
    analysis['sentence_count'] = len([s for s in sentences if s.strip()])
    
    # Sentiment analysis using NLTK Vader
    sentiment_scores = sia.polarity_scores(text)
    analysis['sentiment'] = sentiment_scores['compound']
    analysis['positive_score'] = sentiment_scores['pos']
    analysis['negative_score'] = sentiment_scores['neg']
    analysis['neutral_score'] = sentiment_scores['neu']
    
    # Keyword analysis
    skills_keywords = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node',
        'machine learning', 'data analysis', 'project management', 'agile',
        'communication', 'leadership', 'teamwork', 'problem solving',
        'excel', 'word', 'powerpoint', 'management', 'development',
        'programming', 'coding', 'design', 'analysis', 'research',
        'database', 'api', 'web', 'software', 'technical', 'cloud'
    ]
    
    found_keywords = []
    for keyword in skills_keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', text.lower()):
            found_keywords.append(keyword.title())
    
    analysis['found_keywords'] = found_keywords
    analysis['keyword_count'] = len(found_keywords)
    
    # Section detection
    sections = ['experience', 'education', 'skills', 'projects', 'certifications', 'summary', 'objective', 'contact']
    found_sections = []
    for section in sections:
        if re.search(rf'\b{section}\b', text.lower()):
            found_sections.append(section.title())
    
    analysis['found_sections'] = found_sections
    
    # Contact info detection
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    linkedin_pattern = r'linkedin\.com/in/[A-Za-z0-9-]+'
    
    analysis['has_email'] = bool(re.search(email_pattern, text))
    analysis['has_phone'] = bool(re.search(phone_pattern, text))
    analysis['has_linkedin'] = bool(re.search(linkedin_pattern, text.lower()))
    
    # Action verbs detection
    action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'improved', 
                   'increased', 'reduced', 'achieved', 'built', 'designed', 'coordinated']
    found_verbs = []
    for verb in action_verbs:
        if re.search(rf'\b{verb}\b', text.lower()):
            found_verbs.append(verb.title())
    
    analysis['action_verbs'] = found_verbs
    
    return analysis

# Home Page
if app_mode == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎉 Welcome to ResumePolish!
        
        **Your intelligent resume analysis tool** that helps you create the perfect resume that stands out to employers and ATS systems!
        """)
        
        # Animated features section
        st.markdown("### ✨ Key Features")
        
        features = [
            {"icon": "📊", "title": "Deep Content Analysis", "desc": "Comprehensive analysis of your resume content, keywords, and structure"},
            {"icon": "🎯", "title": "ATS Optimization", "desc": "Improve compatibility with Applicant Tracking Systems"},
            {"icon": "📈", "title": "Visual Analytics", "desc": "Beautiful charts and metrics to understand your resume better"},
            {"icon": "💡", "title": "Smart Suggestions", "desc": "Personalized recommendations to enhance your resume"},
            {"icon": "🔍", "title": "Keyword Analysis", "desc": "Identify missing skills and optimize keyword usage"},
            {"icon": "⭐", "title": "Scoring System", "desc": "Get an overall score and detailed breakdown"}
        ]
        
        # Display features in a grid
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="feature-card">
                    <h3 style="color: #2E86AB; margin-bottom: 10px;">{feature['icon']} {feature['title']}</h3>
                    <p style="color: #555;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Quick start guide
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #2E86AB;">🚀 Quick Start</h3>
            <ol style="color: #555;">
                <li><b>Upload</b> your resume (PDF/DOCX)</li>
                <li><b>Analyze</b> the results</li>
                <li><b>Improve</b> with suggestions</li>
                <li><b>Download</b> better resume!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Why choose us
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #2E86AB;">📊 Why Choose Us?</h3>
            <ul style="color: #555;">
                <li>✅ <b>100% Free</b></li>
                <li>✅ <b>Instant Results</b></li>
                <li>✅ <b>Easy to Use</b></li>
                <li>✅ <b>Professional Tips</b></li>
                <li>✅ <b>No Registration</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Upload Resume Page
elif app_mode == "📤 Upload Resume":
    st.markdown('<div class="sub-header">📤 Upload Your Resume</div>', unsafe_allow_html=True)
    
    # File upload section with nice styling
    st.markdown("""
    <div class="upload-area">
        <h3 style="color: #2E86AB;">📁 Drag and drop your resume here</h3>
        <p style="color: #666;">Supported formats: PDF, DOCX</p>
        <p style="color: #888; font-size: 14px;">We'll analyze your resume and provide detailed feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your resume file", 
        type=['pdf', 'docx'],
        help="Upload your resume in PDF or Word format",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Show file info in a nice card
        file_details = {
            "📄 Filename": uploaded_file.name,
            "💾 File size": f"{uploaded_file.size / 1024:.1f} KB",
            "🔍 File type": uploaded_file.type,
            "⏰ Upload time": datetime.now().strftime("%H:%M:%S")
        }
        
        st.markdown("### 📋 File Details")
        col1, col2 = st.columns(2)
        with col1:
            for key, value in list(file_details.items())[:2]:
                st.info(f"**{key}**: {value}")
        with col2:
            for key, value in list(file_details.items())[2:]:
                st.info(f"**{key}**: {value}")
        
        with st.spinner('🔍 Analyzing your resume... This may take a few seconds.'):
            # Extract text
            resume_text = extract_text_from_file(uploaded_file)
            
            if resume_text and len(resume_text.strip()) > 50:
                # Analyze resume
                analysis_results = analyze_resume(resume_text)
                
                # Store in session state
                st.session_state['resume_text'] = resume_text
                st.session_state['analysis_results'] = analysis_results
                st.session_state['uploaded_file_name'] = uploaded_file.name
                
                st.success("✅ Resume analyzed successfully!")
                st.balloons()
                
                # Show preview in expandable section
                with st.expander("📋 Preview Extracted Text", expanded=False):
                    st.text_area("", resume_text, height=200, key="text_preview")
                    
            else:
                st.error("❌ Could not extract sufficient text from the file. Please try a different file.")

# Analysis Results Page
elif app_mode == "📊 Analysis Results":
    if 'analysis_results' not in st.session_state:
        st.warning("""
        ⚠️ **Please upload your resume first!** 
        
        Go to the **📤 Upload Resume** section to get started.
        """)
    else:
        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        analysis = st.session_state['analysis_results']
        resume_text = st.session_state['resume_text']
        
        # Overall score first
        st.markdown("### 🏆 Overall Resume Score")
        
        # Calculate comprehensive score
        score = 0
        max_score = 100
        
        # Word count score (ideal: 400-600 words)
        if 400 <= analysis['word_count'] <= 600:
            score += 20
        elif 300 <= analysis['word_count'] < 400 or 600 < analysis['word_count'] <= 800:
            score += 15
        else:
            score += 5
        
        # Keywords score
        score += min(analysis['keyword_count'] * 2, 20)
        
        # Sections score
        score += min(len(analysis['found_sections']) * 4, 20)
        
        # Contact info score
        contact_score = 0
        if analysis['has_email']:
            contact_score += 5
        if analysis['has_phone']:
            contact_score += 5
        if analysis['has_linkedin']:
            contact_score += 5
        score += contact_score
        
        # Action verbs score
        score += min(len(analysis['action_verbs']) * 2, 10)
        
        # Sentiment score
        if analysis['sentiment'] > 0.1:
            score += 10
        elif analysis['sentiment'] > 0:
            score += 5
        
        # Display score with visual progress
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Color based on score
            if score >= 80:
                color = "#27ae60"
                emoji = "🎉"
            elif score >= 60:
                color = "#f39c12"
                emoji = "👍"
            else:
                color = "#e74c3c"
                emoji = "📝"
                
            st.markdown(f"<h1 style='text-align: center; color: {color};'>{score}/100 {emoji}</h1>", unsafe_allow_html=True)
            st.progress(score / 100)
            
            if score >= 80:
                st.success("**Excellent!** Your resume is well-optimized and professional!")
            elif score >= 60:
                st.warning("**Good!** Your resume is decent but has room for improvement.")
            else:
                st.error("**Needs work.** Check the suggestions below to improve your resume.")
        
        st.markdown("---")
        
        # Key Metrics in beautiful cards
        st.markdown("### 📈 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Optimal" if 400 <= analysis['word_count'] <= 600 else "⚠️ Check"
            color = "#27ae60" if 400 <= analysis['word_count'] <= 600 else "#f39c12"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{analysis['word_count']}</h3>
                <p>Word Count</p>
                <small style="color: {color};">{status}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            status = "✅ Good" if analysis['keyword_count'] >= 8 else "⚠️ Add more"
            color = "#27ae60" if analysis['keyword_count'] >= 8 else "#f39c12"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{analysis['keyword_count']}</h3>
                <p>Keywords Found</p>
                <small style="color: {color};">{status}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            status = "✅ Good" if len(analysis['found_sections']) >= 4 else "⚠️ Add sections"
            color = "#27ae60" if len(analysis['found_sections']) >= 4 else "#f39c12"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(analysis['found_sections'])}</h3>
                <p>Sections</p>
                <small style="color: {color};">{status}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            sentiment_label = "😊 Positive" if analysis['sentiment'] > 0.1 else "😐 Neutral" if analysis['sentiment'] > -0.1 else "😞 Negative"
            color = "#27ae60" if analysis['sentiment'] > 0.1 else "#f39c12" if analysis['sentiment'] > -0.1 else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{sentiment_label}</h3>
                <p>Tone</p>
                <small style="color: {color};">Based on language</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Analysis
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Keywords visualization
            st.markdown("### 🔑 Skills Keywords Found")
            if analysis['found_keywords']:
                # Create bar chart
                keywords_df = pd.DataFrame({
                    'Keyword': analysis['found_keywords'],
                    'Count': [1] * len(analysis['found_keywords'])
                })
                
                # Plot using matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Blues(np.linspace(0.5, 1, len(keywords_df)))
                bars = ax.barh(keywords_df['Keyword'], keywords_df['Count'], color=colors)
                ax.set_xlabel('Presence')
                ax.set_title('Skills Found in Your Resume')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No common skills keywords found. Consider adding relevant technical and soft skills.")
            
            # Action verbs
            st.markdown("### ⚡ Action Verbs Used")
            if analysis['action_verbs']:
                st.success(f"Found {len(analysis['action_verbs'])} action verbs: {', '.join(analysis['action_verbs'])}")
            else:
                st.warning("No strong action verbs found. Consider adding verbs like 'Managed', 'Developed', 'Created'.")
        
        with col2:
            # Sections analysis
            st.markdown("### 📑 Resume Sections")
            sections_data = {
                'Section': ['Experience', 'Education', 'Skills', 'Projects', 'Certifications', 'Summary'],
                'Found': [
                    1 if 'Experience' in analysis['found_sections'] else 0,
                    1 if 'Education' in analysis['found_sections'] else 0,
                    1 if 'Skills' in analysis['found_sections'] else 0,
                    1 if 'Projects' in analysis['found_sections'] else 0,
                    1 if 'Certifications' in analysis['found_sections'] else 0,
                    1 if 'Summary' in analysis['found_sections'] or 'Objective' in analysis['found_sections'] else 0
                ]
            }
            sections_df = pd.DataFrame(sections_data)
            
            # Plot sections
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#27ae60' if found else '#e74c3c' for found in sections_df['Found']]
            bars = ax.barh(sections_df['Section'], sections_df['Found'], color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Presence (1 = Found, 0 = Missing)')
            ax.set_title('Resume Sections Analysis')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Contact information
            st.markdown("### 📞 Contact Information")
            contact_cols = st.columns(3)
            with contact_cols[0]:
                st.metric("Email", "✅ Found" if analysis['has_email'] else "❌ Missing")
            with contact_cols[1]:
                st.metric("Phone", "✅ Found" if analysis['has_phone'] else "❌ Missing")
            with contact_cols[2]:
                st.metric("LinkedIn", "✅ Found" if analysis['has_linkedin'] else "❌ Missing")

# Tips & Suggestions Page
elif app_mode == "💡 Tips & Suggestions":
    st.markdown('<div class="sub-header">💡 Personalized Tips & Suggestions</div>', unsafe_allow_html=True)
    
    if 'analysis_results' not in st.session_state:
        st.warning("""
        📤 **Please upload your resume first to get personalized suggestions!**
        
        Go to the **📤 Upload Resume** section to analyze your resume.
        """)
        
        # General tips for all users
        st.markdown("### 📝 General Resume Best Practices")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
            **💼 Content Tips:**
            - Use strong action verbs (Managed, Developed, Implemented)
            - Quantify achievements with numbers and metrics
            - Keep length between 300-600 words
            - Include relevant keywords from job descriptions
            
            **🎨 Formatting Tips:**
            - Use clear section headings
            - Consistent formatting throughout
            - Professional fonts (Arial, Calibri, Times New Roman)
            - Adequate white space for readability
            """)
        
        with tips_col2:
            st.markdown("""
            **🔍 ATS Optimization:**
            - Use standard section names
            - Avoid images, tables, and columns
            - Include relevant skills throughout
            - Use .docx format for best compatibility
            
            **🚀 Career Advancement:**
            - Tailor resume for each job application
            - Include most recent experiences first
            - Highlight transferable skills
            - Get feedback from mentors
            """)
    else:
        analysis = st.session_state['analysis_results']
        
        # Personalized suggestions based on analysis
        st.markdown("### 🎯 Personalized Recommendations")
        
        suggestions = []
        
        # Word count suggestions
        if analysis['word_count'] < 300:
            suggestions.append("""
            <div class="improvement-feedback">
            **📝 Add More Content**: Your resume is too brief (less than 300 words). 
            - Add more details about your experiences and responsibilities
            - Include specific achievements with numbers
            - Consider adding projects or certifications sections
            </div>
            """)
        elif analysis['word_count'] > 800:
            suggestions.append("""
            <div class="improvement-feedback">
            **✂️ Reduce Length**: Your resume is too long (over 800 words).
            - Focus on most relevant information
            - Remove outdated or less relevant experiences
            - Use bullet points instead of paragraphs
            - Keep only the last 10-15 years of experience
            </div>
            """)
        else:
            suggestions.append("""
            <div class="positive-feedback">
            **📝 Good Length**: Your resume has an appropriate word count. Keep it up!
            </div>
            """)
        
        # Keyword suggestions
        if analysis['keyword_count'] < 5:
            suggestions.append("""
            <div class="improvement-feedback">
            **🔑 Add More Keywords**: Only found {} relevant keywords.
            - Include more industry-specific technical skills
            - Add soft skills like communication, leadership
            - Mention specific tools and technologies you've used
            - Incorporate action verbs throughout
            </div>
            """.format(analysis['keyword_count']))
        else:
            suggestions.append("""
            <div class="positive-feedback">
            **🔑 Good Keyword Usage**: Found {} relevant keywords. Well done!
            </div>
            """.format(analysis['keyword_count']))
        
        # Section suggestions
        essential_sections = ['Experience', 'Education', 'Skills']
        missing_essential = [section for section in essential_sections if section not in analysis['found_sections']]
        if missing_essential:
            suggestions.append("""
            <div class="improvement-feedback">
            **📑 Add Essential Sections**: Missing {} section(s).
            - These are crucial for a complete resume
            - Use standard section headings
            - Ensure clear organization
            </div>
            """.format(', '.join(missing_essential)))
        
        # Contact info suggestions
        contact_missing = []
        if not analysis['has_email']:
            contact_missing.append("email")
        if not analysis['has_phone']:
            contact_missing.append("phone")
        if not analysis['has_linkedin']:
            contact_missing.append("LinkedIn")
            
        if contact_missing:
            suggestions.append("""
            <div class="improvement-feedback">
            **📞 Add Contact Information**: Missing {}.
            - Include professional email and phone
            - Add LinkedIn profile URL if available
            - Make it easy for employers to contact you
            </div>
            """.format(', '.join(contact_missing)))
        
        # Action verbs suggestions
        if len(analysis['action_verbs']) < 3:
            suggestions.append("""
            <div class="improvement-feedback">
            **⚡ Use More Action Verbs**: Only found {} strong action verbs.
            - Start bullet points with verbs like: Managed, Developed, Created
            - Use past tense for previous roles
            - Show impact and responsibility
            </div>
            """.format(len(analysis['action_verbs'])))
        
        # Display all suggestions
        for suggestion in suggestions:
            st.markdown(suggestion, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 30px; background: #f8f9fa; border-radius: 10px;'>
        <h3 style='color: #2E86AB;'>ResumePolish 📄</h3>
        <p>Your AI-Powered Resume Assistant | Made with ❤️ using Streamlit</p>
        <p>© 2024 ResumePolish. Helping job seekers create better resumes.</p>
    </div>
    """,
    unsafe_allow_html=True
)
