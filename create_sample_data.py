"""
Create sample labeled data for Sprint 0 validation
Generates 20 resumes and 20 job descriptions with labels
"""

import pandas as pd
import json
import os

# Sample resumes (simplified for Sprint 0)
resumes = [
    # Good matches (10 senior DS roles)
    {"id": "R001", "text": "Data Scientist with 5 years experience in Python, ML, deep learning, NLP, and cloud deployment. Built recommendation systems using TensorFlow.", "exp_years": 5, "seniority": "senior"},
    {"id": "R002", "text": "Senior ML Engineer, 6 years exp. Expert in Python, scikit-learn, PyTorch, AWS. Led team of 3 in building fraud detection models.", "exp_years": 6, "seniority": "senior"},
    {"id": "R003", "text": "AI Research Scientist, PhD, 7 years. Published papers on transformer models. Skilled in Python, NLP, computer vision, and model optimization.", "exp_years": 7, "seniority": "senior"},
    {"id": "R004", "text": "Lead Data Scientist at tech startup. 5+ years building production ML pipelines. Python, Spark, Kubernetes, MLflow experience.", "exp_years": 5, "seniority": "senior"},
    {"id": "R005", "text": "Data Science Manager with 8 years experience. Expertise in statistical modeling, A/B testing, Python, R, SQL. Managed DS teams.", "exp_years": 8, "seniority": "senior"},
    
    # Good matches (5 mid-level DS roles)
    {"id": "R006", "text": "Data Scientist, 3 years experience in predictive modeling, Python, pandas, scikit-learn. Built customer churn models.", "exp_years": 3, "seniority": "mid"},
    {"id": "R007", "text": "ML Engineer with 3.5 years exp. Proficient in Python, TensorFlow, SQL, Docker. Deployed multiple ML services in production.", "exp_years": 3, "seniority": "mid"},
    {"id": "R008", "text": "Data Analyst transitioning to DS role. 4 years experience with Python, statistical analysis, dashboarding, and basic ML.", "exp_years": 4, "seniority": "mid"},
    {"id": "R009", "text": "Junior Data Scientist, 2.5 years. Skills: Python, ML algorithms, data visualization, API development. Quick learner.", "exp_years": 2, "seniority": "mid"},
    {"id": "R010", "text": "Machine Learning Engineer, 3 years. Built NLP models for text classification. Python, spaCy, BERT, FastAPI.", "exp_years": 3, "seniority": "mid"},
    
    # Good matches (5 junior DS roles)
    {"id": "R011", "text": "Recent graduate with internship in data science. Python, pandas, matplotlib, basic ML knowledge. Eager to learn.", "exp_years": 0, "seniority": "junior"},
    {"id": "R012", "text": "Entry-level Data Scientist. Completed online courses in ML and deep learning. Projects in image classification and NLP.", "exp_years": 1, "seniority": "junior"},
    {"id": "R013", "text": "Fresher with MS in Data Science. Academic projects in regression, classification, clustering. Python, R, Tableau.", "exp_years": 0, "seniority": "junior"},
    {"id": "R014", "text": "Junior analyst with 1 year experience. Looking to move into ML role. Python, SQL, basic statistics, exploratory data analysis.", "exp_years": 1, "seniority": "junior"},
    {"id": "R015", "text": "Data Science intern, 6 months experience. Worked on time series forecasting project. Python, pandas, scikit-learn.", "exp_years": 0, "seniority": "junior"},
    
    # Poor matches (5 - wrong domain or insufficient experience)
    {"id": "R016", "text": "Software Developer, 4 years Java and Spring Boot. No ML experience but interested in learning data science.", "exp_years": 4, "seniority": "mid"},
    {"id": "R017", "text": "Business Analyst with 3 years experience in Excel, PowerPoint, and requirements gathering. Basic SQL knowledge.", "exp_years": 3, "seniority": "mid"},
    {"id": "R018", "text": "Frontend Developer, 5 years React and JavaScript. Built dashboards but no Python or ML background.", "exp_years": 5, "seniority": "senior"},
    {"id": "R019", "text": "Network Engineer with 6 years experience in Cisco, routing, firewall config. No programming or data analysis skills.", "exp_years": 6, "seniority": "senior"},
    {"id": "R020", "text": "MBA graduate with marketing background. Interested in analytics but no technical skills in Python or statistics.", "exp_years": 2, "seniority": "junior"},
]

# Sample job descriptions
job_descriptions = [
    # Senior roles (5)
    {"id": "JD001", "role": "Senior Data Scientist", "company": "TechCorp", "location": "Bangalore", "exp_min": 5, "exp_max": 8, "seniority": "senior",
     "text": "Looking for Senior Data Scientist with 5+ years experience. Must have strong Python, ML, deep learning skills. Experience with NLP and deployment required."},
    
    {"id": "JD002", "role": "Lead ML Engineer", "company": "FinTech Inc", "location": "Mumbai", "exp_min": 5, "exp_max": 10, "seniority": "senior",
     "text": "Lead ML Engineer needed. 5+ years building production ML systems. Python, PyTorch, AWS, Kubernetes expertise required. Leadership experience preferred."},
    
    {"id": "JD003", "role": "Data Science Manager", "company": "E-commerce Giant", "location": "Hyderabad", "exp_min": 6, "exp_max": 10, "seniority": "senior",
     "text": "Data Science Manager position. 6+ years in DS/ML. Manage team, design experiments, statistical modeling. Python, R, SQL required."},
    
    {"id": "JD004", "role": "Senior AI Engineer", "company": "AI Startup", "location": "Pune", "exp_min": 5, "exp_max": 8, "seniority": "senior",
     "text": "Senior AI Engineer role. 5+ years NLP/computer vision experience. Transformer models, model optimization, Python, TensorFlow/PyTorch required."},
    
    {"id": "JD005", "role": "Principal Data Scientist", "company": "Consulting Firm", "location": "Delhi", "exp_min": 7, "exp_max": 12, "seniority": "senior",
     "text": "Principal DS position. 7+ years advanced analytics, ML model development. Client-facing role. Python, Spark, cloud platforms essential."},
    
    # Mid-level roles (10)
    {"id": "JD006", "role": "Data Scientist", "company": "Healthcare Tech", "location": "Bangalore", "exp_min": 2, "exp_max": 5, "seniority": "mid",
     "text": "Data Scientist with 2-5 years experience. Build predictive models, work with medical data. Python, scikit-learn, SQL required."},
    
    {"id": "JD007", "role": "ML Engineer", "company": "EdTech Startup", "location": "Mumbai", "exp_min": 3, "exp_max": 5, "seniority": "mid",
     "text": "ML Engineer needed. 3-5 years experience deploying ML models. Python, TensorFlow, Docker, REST APIs. Education domain experience a plus."},
    
    {"id": "JD008", "role": "Data Scientist", "company": "Retail Chain", "location": "Bangalore", "exp_min": 2, "exp_max": 4, "seniority": "mid",
     "text": "DS role for retail analytics. 2-4 years experience in customer analytics, demand forecasting. Python, pandas, visualization tools."},
    
    {"id": "JD009", "role": "Machine Learning Engineer", "company": "SaaS Company", "location": "Hyderabad", "exp_min": 3, "exp_max": 5, "seniority": "mid",
     "text": "ML Engineer position. 3-5 years building recommendation systems or NLP models. Python, ML libraries, API development skills needed."},
    
    {"id": "JD010", "role": "Data Scientist", "company": "Gaming Company", "location": "Pune", "exp_min": 2, "exp_max": 5, "seniority": "mid",
     "text": "Data Scientist for gaming analytics. 2-5 years experience. Player behavior analysis, ML models. Python, SQL, big data tools."},
    
    {"id": "JD011", "role": "Applied Scientist", "company": "Search Engine Co", "location": "Bangalore", "exp_min": 3, "exp_max": 6, "seniority": "mid",
     "text": "Applied Scientist role. 3-6 years ML/NLP experience. Ranking algorithms, information retrieval. Python, deep learning frameworks."},
    
    {"id": "JD012", "role": "Data Scientist", "company": "Insurance Firm", "location": "Mumbai", "exp_min": 2, "exp_max": 5, "seniority": "mid",
     "text": "DS for insurance analytics. 2-5 years. Risk modeling, claims prediction. Python, statistical modeling, SQL essential."},
    
    {"id": "JD013", "role": "ML Engineer", "company": "Logistics Startup", "location": "Delhi", "exp_min": 3, "exp_max": 5, "seniority": "mid",
     "text": "ML Engineer for route optimization and demand prediction. 3-5 years experience. Python, ML algorithms, deployment experience."},
    
    {"id": "JD014", "role": "Data Scientist", "company": "Media Company", "location": "Bangalore", "exp_min": 2, "exp_max": 4, "seniority": "mid",
     "text": "Data Scientist role. 2-4 years content recommendation and user analytics. Python, ML, A/B testing experience required."},
    
    {"id": "JD015", "role": "ML Engineer", "company": "Fintech Startup", "location": "Hyderabad", "exp_min": 3, "exp_max": 5, "seniority": "mid",
     "text": "ML Engineer for fraud detection systems. 3-5 years. Anomaly detection, real-time ML. Python, Spark, cloud platforms."},
    
    # Junior roles (5)
    {"id": "JD016", "role": "Junior Data Scientist", "company": "Analytics Firm", "location": "Bangalore", "exp_min": 0, "exp_max": 2, "seniority": "junior",
     "text": "Junior Data Scientist position. 0-2 years or fresh graduate. Strong Python, basic ML knowledge. Willingness to learn required."},
    
    {"id": "JD017", "role": "Data Analyst", "company": "E-commerce", "location": "Mumbai", "exp_min": 0, "exp_max": 1, "seniority": "junior",
     "text": "Entry-level Data Analyst role. Fresh graduates welcome. Python, SQL, data visualization. Interest in ML a plus."},
    
    {"id": "JD018", "role": "Associate Data Scientist", "company": "Consulting Co", "location": "Pune", "exp_min": 0, "exp_max": 2, "seniority": "junior",
     "text": "Associate DS role for freshers or 0-2 years exp. Academic ML projects acceptable. Python, pandas, scikit-learn basics needed."},
    
    {"id": "JD019", "role": "ML Intern", "company": "AI Research Lab", "location": "Bangalore", "exp_min": 0, "exp_max": 1, "seniority": "junior",
     "text": "ML internship opportunity. Recent graduates in CS/related fields. Python programming, basic ML algorithms. Research interest preferred."},
    
    {"id": "JD020", "role": "Junior ML Engineer", "company": "Tech Startup", "location": "Hyderabad", "exp_min": 0, "exp_max": 2, "seniority": "junior",
     "text": "Junior ML Engineer position. 0-2 years or strong academic background. Python, ML basics, eagerness to learn production systems."},
]

# Create labeled pairs (good_match and poor_match)
labeled_pairs = []

# Good matches: similar seniority and relevant skills (30 pairs)
# Senior-Senior matches
for i in range(5):
    labeled_pairs.append({
        "resume_id": resumes[i]["id"],
        "jd_id": job_descriptions[i]["id"],
        "resume_text": resumes[i]["text"],
        "jd_text": job_descriptions[i]["text"],
        "resume_exp": resumes[i]["exp_years"],
        "jd_exp_min": job_descriptions[i]["exp_min"],
        "jd_exp_max": job_descriptions[i]["exp_max"],
        "label": "good_match"
    })

# Mid-Mid matches
for i in range(5, 10):
    labeled_pairs.append({
        "resume_id": resumes[i]["id"],
        "jd_id": job_descriptions[i]["id"],
        "resume_text": resumes[i]["text"],
        "jd_text": job_descriptions[i]["text"],
        "resume_exp": resumes[i]["exp_years"],
        "jd_exp_min": job_descriptions[i]["exp_min"],
        "jd_exp_max": job_descriptions[i]["exp_max"],
        "label": "good_match"
    })

# Junior-Junior matches
for i in range(10, 15):
    labeled_pairs.append({
        "resume_id": resumes[i]["id"],
        "jd_id": job_descriptions[15+i-10]["id"],
        "resume_text": resumes[i]["text"],
        "jd_text": job_descriptions[15+i-10]["text"],
        "resume_exp": resumes[i]["exp_years"],
        "jd_exp_min": job_descriptions[15+i-10]["exp_min"],
        "jd_exp_max": job_descriptions[15+i-10]["exp_max"],
        "label": "good_match"
    })

# Poor matches: wrong domain or experience mismatch (10 pairs)
# Wrong domain candidates to senior roles
for i in range(15, 20):
    jd_idx = i - 15
    labeled_pairs.append({
        "resume_id": resumes[i]["id"],
        "jd_id": job_descriptions[jd_idx]["id"],
        "resume_text": resumes[i]["text"],
        "jd_text": job_descriptions[jd_idx]["text"],
        "resume_exp": resumes[i]["exp_years"],
        "jd_exp_min": job_descriptions[jd_idx]["exp_min"],
        "jd_exp_max": job_descriptions[jd_idx]["exp_max"],
        "label": "poor_match"
    })

# Experience mismatch: junior resumes to senior roles
for i in range(10, 15):
    labeled_pairs.append({
        "resume_id": resumes[i]["id"],
        "jd_id": job_descriptions[i-10]["id"],  # Senior JDs
        "resume_text": resumes[i]["text"],
        "jd_text": job_descriptions[i-10]["text"],
        "resume_exp": resumes[i]["exp_years"],
        "jd_exp_min": job_descriptions[i-10]["exp_min"],
        "jd_exp_max": job_descriptions[i-10]["exp_max"],
        "label": "poor_match"
    })

# Save to files
os.makedirs("data/samples", exist_ok=True)

# Save labeled pairs
df = pd.DataFrame(labeled_pairs)
df.to_csv("data/samples/labeled_pairs.csv", index=False)

# Save separate resume and JD files for reference
pd.DataFrame(resumes).to_csv("data/samples/resumes.csv", index=False)
pd.DataFrame(job_descriptions).to_csv("data/samples/job_descriptions.csv", index=False)

# Print statistics
print("Sample data created successfully!")
print(f"\nTotal labeled pairs: {len(labeled_pairs)}")
print(f"Good matches: {sum(1 for p in labeled_pairs if p['label'] == 'good_match')}")
print(f"Poor matches: {sum(1 for p in labeled_pairs if p['label'] == 'poor_match')}")
print(f"\nFiles saved to data/samples/")
print("- labeled_pairs.csv")
print("- resumes.csv")
print("- job_descriptions.csv")