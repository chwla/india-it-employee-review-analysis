import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import os

st.set_page_config(page_title="ATS Resume Matcher", page_icon="🎯", layout="wide")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_resources():
    status = st.empty()
    status.text("Loading AI models...")
    
    errors = []
    
    # Load TF-IDF components
    try:
        df_naukri = pd.read_csv('data/processed/naukri_processed.csv').fillna("")
        tfidf = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf.fit(df_naukri['clean_text'])
        
        # Load threshold if available
        if os.path.exists('models/tfidf_threshold.txt'):
            with open('models/tfidf_threshold.txt', 'r') as f:
                tfidf_threshold = float(f.read().strip())
        else:
            tfidf_threshold = 0.5
            
    except Exception as e:
        errors.append(f"TF-IDF: {str(e)}")
        tfidf = None
        tfidf_threshold = 0.5

    # Load LSTM components
    lstm_model = None
    tokenizer = None
    
    # Try new Keras format first
    if os.path.exists('models/lstm_model.keras'):
        try:
            lstm_model = load_model('models/lstm_model.keras')
            with open('models/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except Exception as e:
            errors.append(f"LSTM (.keras): {str(e)}")
            lstm_model = None
    
    # Fallback to .h5 format
    if lstm_model is None and os.path.exists('models/lstm_model.h5'):
        try:
            lstm_model = load_model('models/lstm_model.h5', compile=False)
            lstm_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            with open('models/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except Exception as e:
            errors.append(f"LSTM (.h5): {str(e)}")
            lstm_model = None
    
    if lstm_model is None and tokenizer is None:
        errors.append("LSTM: No valid model file found. Run training first.")
        
    status.empty()
    
    if errors:
        st.sidebar.warning("Some models failed to load:\n\n" + "\n\n".join(errors))
    
    return tfidf, tfidf_threshold, lstm_model, tokenizer

def get_skill_overlap(resume, jd):
    """Extract overlapping technical terms"""
    resume_words = set(resume.lower().split())
    jd_words = set(jd.lower().split())
    
    # Filter for meaningful technical terms
    technical_terms = resume_words & jd_words
    technical_terms = {w for w in technical_terms if len(w) > 3}
    
    return list(technical_terms)[:10]  # Top 10

def get_interpretation(score, model_name):
    """Provide human-readable interpretation"""
    if model_name == "TF-IDF":
        if score > 0.7:
            return "🟢 Strong Match", "Keywords align exceptionally well. High probability of ATS clearance."
        elif score > 0.5:
            return "🟡 Moderate Match", "Decent keyword overlap. Consider adding more relevant skills."
        elif score > 0.3:
            return "🟠 Weak Match", "Limited overlap detected. Resume needs significant optimization."
        else:
            return "🔴 Poor Match", "Very low keyword alignment. Major mismatch with job requirements."
    else:  # LSTM
        if score > 0.7:
            return "🟢 Strong Semantic Match", "Deep learning model detects strong contextual alignment."
        elif score > 0.5:
            return "🟡 Moderate Match", "Some semantic similarity detected."
        elif score > 0.3:
            return "🟠 Weak Match", "Limited semantic alignment."
        else:
            return "🔴 Poor Match", "Weak semantic relationship detected."

def main():
    st.title("ATS Resume Matcher AI")
    st.markdown("### Analyze Resume-Job Match using Dual AI Models")
    st.markdown("---")
    
    # Load models
    tfidf_vectorizer, tfidf_threshold, lstm_model, tokenizer = load_resources()
    
    if not tfidf_vectorizer and not lstm_model:
        st.error("No models available. Please run the training pipeline first.")
        st.code("""
# Run these commands in order:
python src/preprocessing.py
python src/build_dataset.py
python src/baseline_tfidf.py
python src/train_lstm.py
        """)
        return

    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool uses two AI approaches:
        
        **1. TF-IDF Baseline** 
        - Keyword-based matching
        - Fast and interpretable
        - Mimics traditional ATS systems
        
        **2. LSTM Siamese Network** 
        - Deep learning approach
        - Captures semantic meaning
        - Understands context beyond keywords
        
        **How to use:**
        1. Paste resume text (left)
        2. Paste job description (right)
        3. Click "Analyze Match"
        """)
        
        st.markdown("---")
        
        # Model status
        st.subheader("Model Status")
        if tfidf_vectorizer:
            st.success("TF-IDF Model loaded")
        else:
            st.error("TF-IDF Model unavailable")
            
        if lstm_model:
            st.success("LSTM Model loaded")
        else:
            st.error("LSTM Model unavailable")
        
        st.markdown("---")
        st.caption("Built with TensorFlow, scikit-learn, and Streamlit")
    
    # Input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resume")
        resume_text = st.text_area(
            "Paste Resume Text",
            height=350,
            placeholder="Experienced Data Scientist with 5 years in ML/AI...\n\nSkills: Python, TensorFlow, AWS...",
            help="Paste the complete resume text here"
        )
        
    with col2:
        st.subheader("Job Description")
        jd_text = st.text_area(
            "Paste Job Description",
            height=350,
            placeholder="Looking for Senior Data Scientist...\n\nRequired: Python, ML, Cloud...",
            help="Paste the complete job description here"
        )

    # Analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_btn = st.button("🔍 Analyze Match", type="primary", use_container_width=True)

    if analyze_btn:
        if not resume_text.strip() or not jd_text.strip():
            st.warning("Please provide both resume and job description.")
            return
            
        with st.spinner("Analyzing match..."):
            # Preprocess
            clean_resume = clean_text(resume_text)
            clean_jd = clean_text(jd_text)
            
            if len(clean_resume) < 20 or len(clean_jd) < 20:
                st.error("Text too short. Please provide more content.")
                return
            
            results = {}
            
            # --- TF-IDF Model ---
            if tfidf_vectorizer:
                try:
                    vec_resume = tfidf_vectorizer.transform([clean_resume])
                    vec_jd = tfidf_vectorizer.transform([clean_jd])
                    cosine_sim = cosine_similarity(vec_resume, vec_jd)[0][0]
                    results['tfidf_score'] = round(cosine_sim * 100, 2)
                    results['tfidf_raw'] = cosine_sim
                except Exception as e:
                    st.error(f"TF-IDF error: {e}")
            
            # --- LSTM Model ---
            if lstm_model and tokenizer:
                try:
                    seq_resume = tokenizer.texts_to_sequences([clean_resume])
                    pad_resume = pad_sequences(seq_resume, maxlen=150, padding='post', truncating='post')
                    
                    seq_jd = tokenizer.texts_to_sequences([clean_jd])
                    pad_jd = pad_sequences(seq_jd, maxlen=150, padding='post', truncating='post')
                    
                    lstm_pred = lstm_model.predict([pad_resume, pad_jd], verbose=0)[0][0]
                    results['lstm_score'] = round(lstm_pred * 100, 2)
                    results['lstm_raw'] = lstm_pred
                except Exception as e:
                    st.error(f"LSTM error: {e}")

        # --- Display Results ---
        st.markdown("---")
        st.header("Match Analysis Results")
        
        if 'tfidf_score' in results or 'lstm_score' in results:
            # Metric cards
            metric_cols = st.columns(2)
            
            if 'tfidf_score' in results:
                with metric_cols[0]:
                    st.metric(
                        label="TF-IDF Match Score",
                        value=f"{results['tfidf_score']}%",
                        help="Keyword-based similarity score"
                    )
                    status, explanation = get_interpretation(results['tfidf_raw'], "TF-IDF")
                    st.markdown(f"**{status}**")
                    st.caption(explanation)
            
            if 'lstm_score' in results:
                with metric_cols[1]:
                    st.metric(
                        label="LSTM Match Score",
                        value=f"{results['lstm_score']}%",
                        help="Semantic similarity score"
                    )
                    status, explanation = get_interpretation(results['lstm_raw'], "LSTM")
                    st.markdown(f"**{status}**")
                    st.caption(explanation)
            
            # Skill overlap analysis
            st.markdown("---")
            st.subheader("Key Skill Overlap")
            skills = get_skill_overlap(resume_text, jd_text)
            
            if skills:
                skill_cols = st.columns(5)
                for idx, skill in enumerate(skills):
                    with skill_cols[idx % 5]:
                        st.markdown(f" `{skill}`")
            else:
                st.info("No common technical terms detected. This may indicate a poor match.")
            
            # Recommendations
            st.markdown("---")
            st.subheader("Recommendations")
            
            avg_score = np.mean([results.get('tfidf_raw', 0), results.get('lstm_raw', 0)])
            
            if avg_score > 0.65:
                st.success("""
                **Strong Match Detected**
                - Your resume aligns well with the job requirements
                - Consider highlighting specific achievements that match the JD
                - You have a good chance of passing ATS screening
                """)
            elif avg_score > 0.4:
                st.warning("""
                **Moderate Match - Optimization Needed**
                - Add more keywords from the job description
                - Rephrase experiences to match JD terminology
                - Highlight relevant projects and technologies
                """)
            else:
                st.error("""
                **Weak Match - Major Changes Required**
                - Significant skill gap detected
                - Consider if this role truly matches your profile
                - If applying, heavily customize your resume for this position
                """)
        else:
            st.error("Failed to generate predictions. Check model availability.")

if __name__ == "__main__":
    main()
