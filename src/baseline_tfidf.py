import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import os

def run_baseline():
    labeled_path = 'data/processed/train_data_large.csv'
    naukri_path = 'data/processed/naukri_processed.csv'
    
    print("Loading data...")
    if not os.path.exists(labeled_path):
        print(f"Error: {labeled_path} not found. Run build_dataset.py first.")
        return

    df_labeled = pd.read_csv(labeled_path)
    df_labeled['clean_resume'] = df_labeled['clean_resume'].fillna('')
    df_labeled['clean_jd'] = df_labeled['clean_jd'].fillna('')
    
    # Load Naukri data for vocabulary
    df_naukri = pd.read_csv(naukri_path)
    df_naukri['clean_text'] = df_naukri['clean_text'].fillna('')

    # --- Vectorization ---
    print("Training TF-IDF vectorizer on industry corpus...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better matching
        min_df=2
    )
    vectorizer.fit(df_naukri['clean_text'])
    
    print(f"Transforming {len(df_labeled)} pairs...")
    tfidf_resumes = vectorizer.transform(df_labeled['clean_resume'])
    tfidf_jds = vectorizer.transform(df_labeled['clean_jd'])
    
    # --- Similarity Calculation ---
    cosine_dists = paired_cosine_distances(tfidf_resumes, tfidf_jds)
    similarities = 1 - cosine_dists
    
    df_labeled['tfidf_similarity'] = similarities
    
    # --- Evaluation ---
    df_labeled['target'] = df_labeled['label'].apply(lambda x: 1 if x == 'good_match' else 0)
    
    print("\n" + "="*50)
    print("BASELINE MODEL RESULTS (TF-IDF)")
    print("="*50)
    
    avg_good = df_labeled[df_labeled['target'] == 1]['tfidf_similarity'].mean()
    avg_poor = df_labeled[df_labeled['target'] == 0]['tfidf_similarity'].mean()
    std_good = df_labeled[df_labeled['target'] == 1]['tfidf_similarity'].std()
    std_poor = df_labeled[df_labeled['target'] == 0]['tfidf_similarity'].std()
    
    print(f"\nSimilarity Statistics:")
    print(f"  Good Matches: μ={avg_good:.4f}, σ={std_good:.4f}")
    print(f"  Poor Matches: μ={avg_poor:.4f}, σ={std_poor:.4f}")
    print(f"  Separation Gap: {avg_good - avg_poor:.4f}")
    
    # ROC-AUC
    try:
        auc = roc_auc_score(df_labeled['target'], df_labeled['tfidf_similarity'])
        print(f"\nROC-AUC Score: {auc:.4f}")
    except ValueError as e:
        print(f"ROC-AUC calculation failed: {e}")
        auc = None

    # Dynamic threshold (optimal separation point)
    threshold = (avg_good + avg_poor) / 2
    predictions = (df_labeled['tfidf_similarity'] >= threshold).astype(int)
    
    print(f"\nClassification Results (Threshold = {threshold:.4f}):")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(df_labeled['target'], predictions)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(
        df_labeled['target'], 
        predictions, 
        target_names=['Poor Match', 'Good Match'],
        digits=4
    ))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    out_file = 'results/baseline_predictions.csv'
    df_labeled.to_csv(out_file, index=False)
    print(f"\nPredictions saved to {out_file}")
    
    # Save threshold for dashboard
    with open('models/tfidf_threshold.txt', 'w') as f:
        f.write(str(threshold))
    
    print("="*50)

if __name__ == "__main__":
    run_baseline()
