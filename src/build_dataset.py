import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# --- Config ---
INPUT_FILE = 'data/naukri_data_science_jobs_india.csv'
OUTPUT_FILE = 'data/processed/train_data_large.csv'
SAMPLES_TO_GENERATE = 2000

# Synonym mappings for paraphrasing
SKILL_SYNONYMS = {
    'python': ['python', 'py', 'python programming'],
    'machine learning': ['machine learning', 'ml', 'predictive modeling', 'statistical modeling'],
    'data science': ['data science', 'data analytics', 'analytics'],
    'sql': ['sql', 'database', 'rdbms', 'structured query language'],
    'deep learning': ['deep learning', 'neural networks', 'dl', 'artificial neural networks'],
    'aws': ['aws', 'amazon web services', 'cloud computing'],
    'tableau': ['tableau', 'data visualization', 'bi tools'],
    'spark': ['spark', 'apache spark', 'pyspark'],
    'java': ['java', 'core java', 'java programming'],
    'tensorflow': ['tensorflow', 'tf', 'keras'],
    'hadoop': ['hadoop', 'big data', 'distributed computing'],
    'statistics': ['statistics', 'statistical analysis', 'statistical methods'],
    'r': ['r', 'r programming', 'r language'],
    'excel': ['excel', 'ms excel', 'spreadsheets', 'advanced excel'],
    'power bi': ['power bi', 'powerbi', 'microsoft power bi'],
    'etl': ['etl', 'data pipeline', 'data integration'],
    'docker': ['docker', 'containerization', 'containers'],
    'kubernetes': ['kubernetes', 'k8s', 'orchestration'],
    'git': ['git', 'version control', 'github', 'gitlab'],
    'agile': ['agile', 'scrum', 'agile methodology']
}

# Resume narrative templates with varied structures
RESUME_TEMPLATES = [
    "Experienced {role} with {years} years in {domain}. Expertise in {skills}. Proven track record in {achievement}.",
    "{years}+ years as {role}. Proficient in {skills}. Strong background in {domain} with focus on {achievement}.",
    "Senior professional specializing in {domain}. Core competencies include {skills}. {years} years delivering {achievement}.",
    "{role} with extensive experience in {skills}. {years} years in {domain}. Known for {achievement}.",
    "Skilled {role} | {years} years experience | {skills} | Delivered {achievement} in {domain} environment.",
    "Results-driven {role}. Key skills: {skills}. {years} years expertise in {domain}. Successfully {achievement}.",
]

# Achievement phrases
ACHIEVEMENTS = [
    "model deployment and optimization",
    "scalable data pipelines",
    "business insights from complex datasets",
    "end-to-end ML solutions",
    "data-driven decision making",
    "high-performance analytics platforms",
    "predictive analytics solutions",
    "automated reporting systems",
    "real-time data processing",
    "cost optimization through analytics"
]

ROLES = [
    "Data Scientist", "Machine Learning Engineer", "Data Engineer", 
    "Business Analyst", "Data Analyst", "Senior Data Scientist",
    "ML Engineer", "Analytics Manager", "Big Data Engineer"
]

DOMAINS = [
    "financial services", "healthcare analytics", "e-commerce",
    "technology", "retail analytics", "telecommunications",
    "insurance", "supply chain", "marketing analytics"
]

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\.]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def apply_synonym_variation(text):
    """Apply random synonym substitution to add natural variation"""
    words = text.split()
    for key, synonyms in SKILL_SYNONYMS.items():
        if key in text:
            replacement = np.random.choice(synonyms)
            text = text.replace(key, replacement)
    return text

def extract_key_skills(text, n=5):
    """Extract key skill terms from JD text"""
    words = text.split()
    # Filter for meaningful technical terms (longer than 2 chars)
    skills = [w for w in words if len(w) > 2]
    # Take a random subset
    if len(skills) > n:
        selected = np.random.choice(skills, size=min(n, len(skills)), replace=False)
        return ' '.join(selected)
    return ' '.join(skills[:n])

def generate_realistic_resume(jd_text, match_ratio=0.7):
    """
    Generate a realistic resume from JD with controlled skill overlap
    match_ratio: 0.7 means 70% skills match, 30% different
    """
    # Extract core skills
    skills = extract_key_skills(jd_text, n=7)
    
    # Apply synonym variation to some skills
    skills_varied = apply_synonym_variation(skills)
    
    # Take only a subset of skills (match_ratio controls overlap)
    words = skills_varied.split()
    n_match = int(len(words) * match_ratio)
    matched_skills = ' '.join(np.random.choice(words, size=min(n_match, len(words)), replace=False))
    
    # Generate resume narrative
    template = np.random.choice(RESUME_TEMPLATES)
    resume = template.format(
        role=np.random.choice(ROLES),
        years=np.random.choice([2, 3, 4, 5, 6, 7, 8]),
        domain=np.random.choice(DOMAINS),
        skills=matched_skills,
        achievement=np.random.choice(ACHIEVEMENTS)
    )
    
    return resume

def build_real_dataset():
    print(f"📄 Loading real data from {INPUT_FILE}...")
    try:
        df_raw = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Error: Naukri CSV not found. Please check the data folder.")
        return

    df_raw['text_content'] = df_raw['Job_Role'].fillna('') + " " + df_raw['Skills/Description'].fillna('')
    df_raw['clean_text'] = df_raw['text_content'].apply(clean_text)
    
    df = df_raw[df_raw['clean_text'].str.len() > 20].copy()
    print(f"✅ Loaded {len(df)} valid job descriptions.")
    
    # --- 1. Generate POSITIVE Matches (High similarity but NOT exact copies) ---
    print("🛠 Generating Realistic Positive Samples...")
    
    positives = []
    for _ in range(SAMPLES_TO_GENERATE // 2):
        jd = df.sample(n=1).iloc[0]['clean_text']
        # Generate resume with 70-85% skill overlap (realistic for good matches)
        match_ratio = np.random.uniform(0.70, 0.85)
        resume = generate_realistic_resume(jd, match_ratio=match_ratio)
        
        positives.append({
            'resume_text': resume,
            'jd_text': jd,
            'label': 'good_match'
        })
    
    df_pos = pd.DataFrame(positives)
    
    # --- 2. Generate NEGATIVE Matches (Hard Negatives) ---
    print("🛠 Generating Hard Negative Samples...")
    
    negatives = []
    
    # Pre-calculate vectors for similarity checking
    tfidf = TfidfVectorizer(max_features=1000)
    vectors = tfidf.fit_transform(df['clean_text'])
    
    attempts = 0
    while len(negatives) < (SAMPLES_TO_GENERATE // 2) and attempts < SAMPLES_TO_GENERATE * 5:
        attempts += 1
        
        idx1, idx2 = np.random.choice(df.index, 2, replace=False)
        pos1 = df.index.get_loc(idx1)
        pos2 = df.index.get_loc(idx2)
        
        sim = cosine_similarity(vectors[pos1], vectors[pos2])[0][0]
        
        # Create different types of negatives
        if sim < 0.25:  # Very different roles
            jd_text = df.loc[idx1, 'clean_text']
            # Create resume from completely different skills
            other_jd = df.loc[idx2, 'clean_text']
            resume = generate_realistic_resume(other_jd, match_ratio=0.8)
            
            negatives.append({
                'resume_text': resume,
                'jd_text': jd_text,
                'label': 'poor_match'
            })
        elif 0.25 <= sim < 0.4:  # Semi-hard negatives (some overlap but poor match)
            jd_text = df.loc[idx1, 'clean_text']
            # Generate resume with only 20-40% overlap
            resume = generate_realistic_resume(jd_text, match_ratio=np.random.uniform(0.2, 0.4))
            
            negatives.append({
                'resume_text': resume,
                'jd_text': jd_text,
                'label': 'poor_match'
            })
            
    df_neg = pd.DataFrame(negatives)
    print(f"   Found {len(df_neg)} negative pairs after {attempts} attempts.")

    # --- 3. Combine and Save ---
    df_final = pd.concat([df_pos, df_neg], ignore_index=True)
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    df_final['clean_resume'] = df_final['resume_text'].apply(clean_text)
    df_final['clean_jd'] = df_final['jd_text'].apply(clean_text)
    
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 DATASET CREATED: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df_final)}")
    print(f"Good Matches: {len(df_final[df_final['label']=='good_match'])}")
    print(f"Poor Matches: {len(df_final[df_final['label']=='poor_match'])}")
    print("="*50)

if __name__ == "__main__":
    build_real_dataset()