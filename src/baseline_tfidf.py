import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import os

def run_baseline():
    # File paths
    labeled_path = 'data/processed/labeled_pairs_processed.csv'
    naukri_path = 'data/processed/naukri_processed.csv'
    
    print("Loading data...")
    df_labeled = pd.read_csv(labeled_path)
    # Fill NaN values to avoid errors
    df_labeled['clean_resume'] = df_labeled['clean_resume'].fillna('')
    df_labeled['clean_jd'] = df_labeled['clean_jd'].fillna('')
    
    # Load Naukri data for rich vocabulary
    df_naukri = pd.read_csv(naukri_path)
    df_naukri['clean_text'] = df_naukri['clean_text'].fillna('')

    # --- Step 1: Vectorization ---
    print("Training TF-IDF vectorizer on Naukri corpus (Real-world Vocab)...")
    # We use the large Naukri dataset to learn the vocabulary (IDF part)
    # This helps the model recognize words like 'pyspark' even if they appear rarely in our small labeled set
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectorizer.fit(df_naukri['clean_text'])
    
    print("Transforming labeled Resumes and JDs...")
    tfidf_resumes = vectorizer.transform(df_labeled['clean_resume'])
    tfidf_jds = vectorizer.transform(df_labeled['clean_jd'])
    
    # --- Step 2: Similarity Calculation ---
    # Calculate Cosine Similarity between each Resume and its paired JD
    # Note: paired_cosine_distances returns DISTANCE (0=identical, 1=different)
    # So Similarity = 1 - Distance
    cosine_dists = paired_cosine_distances(tfidf_resumes, tfidf_jds)
    similarities = 1 - cosine_dists
    
    df_labeled['tfidf_similarity'] = similarities
    
    # --- Step 3: Evaluation ---
    # Map text labels to integers: good_match=1, poor_match=0
    df_labeled['target'] = df_labeled['label'].apply(lambda x: 1 if x == 'good_match' else 0)
    
    print("\n" + "="*40)
    print("BASELINE MODEL RESULTS")
    print("="*40)
    
    # 1. Compare Averages
    avg_good = df_labeled[df_labeled['target'] == 1]['tfidf_similarity'].mean()
    avg_poor = df_labeled[df_labeled['target'] == 0]['tfidf_similarity'].mean()
    
    print(f"Avg Similarity (Good Matches): {avg_good:.4f}")
    print(f"Avg Similarity (Poor Matches): {avg_poor:.4f}")
    print(f"Gap: {avg_good - avg_poor:.4f}")
    
    # 2. ROC-AUC Score (How well does the score rank good matches higher than poor ones?)
    try:
        auc = roc_auc_score(df_labeled['target'], df_labeled['tfidf_similarity'])
        print(f"ROC-AUC Score: {auc:.4f}")
    except ValueError:
        print("ROC-AUC requires both classes in the data.")

    # 3. Classification Accuracy (using a dynamic threshold)
    # A simple threshold is the midpoint between the two averages
    threshold = (avg_good + avg_poor) / 2
    predictions = (df_labeled['tfidf_similarity'] >= threshold).astype(int)
    
    print(f"\nClassification Report (Threshold = {threshold:.4f}):")
    print(classification_report(df_labeled['target'], predictions, target_names=['Poor Match', 'Good Match']))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    out_file = 'results/baseline_predictions.csv'
    df_labeled.to_csv(out_file, index=False)
    print(f"predictions saved to {out_file}")

if __name__ == "__main__":
    run_baseline()