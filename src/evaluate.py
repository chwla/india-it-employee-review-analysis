import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import seaborn as sns
import os

def load_predictions():
    base_path = 'results/baseline_predictions.csv'
    lstm_path = 'results/lstm_predictions.csv'
    
    if not os.path.exists(base_path) or not os.path.exists(lstm_path):
        print("Error: Missing prediction files.")
        print("   Run: python baseline_tfidf.py && python train_lstm.py")
        return None

    df_base = pd.read_csv(base_path)
    df_lstm = pd.read_csv(lstm_path)
    
    df_base['lstm_score'] = df_lstm['lstm_score']
    
    return df_base

def calculate_metrics(y_true, y_scores, model_name):
    """Calculate comprehensive metrics with optimal threshold"""
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    return {
        'Model': model_name,
        'AUC': roc_auc_score(y_true, y_scores),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Optimal Threshold': optimal_threshold
    }

def plot_comparison(df):
    """Create comprehensive visualization of model performance"""
    
    fig = plt.figure(figsize=(16, 10))
    
    colors = df['label'].map({'good_match': '#2ecc71', 'poor_match': '#e74c3c'})
    y_true = df['label'].apply(lambda x: 1 if x == 'good_match' else 0).values
    
    # 1. Score Distribution Scatter
    plt.subplot(2, 3, 1)
    plt.scatter(range(len(df)), df['tfidf_similarity'], c=colors, alpha=0.6, s=20)
    plt.axhline(y=df['tfidf_similarity'].mean(), color='blue', linestyle='--', alpha=0.5)
    plt.title("TF-IDF Baseline: Score Distribution", fontsize=12, fontweight='bold')
    plt.ylabel("Similarity Score")
    plt.xlabel("Sample Index")
    plt.grid(True, alpha=0.3)
    plt.legend(['Mean', 'Good Match', 'Poor Match'], loc='upper right')
    
    plt.subplot(2, 3, 2)
    plt.scatter(range(len(df)), df['lstm_score'], c=colors, alpha=0.6, s=20)
    plt.axhline(y=df['lstm_score'].mean(), color='blue', linestyle='--', alpha=0.5)
    plt.title("LSTM: Score Distribution", fontsize=12, fontweight='bold')
    plt.ylabel("Similarity Score")
    plt.xlabel("Sample Index")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 2. Histograms
    plt.subplot(2, 3, 3)
    plt.hist(df[df['label']=='good_match']['tfidf_similarity'], 
             bins=30, alpha=0.7, color='#2ecc71', label='Good Match')
    plt.hist(df[df['label']=='poor_match']['tfidf_similarity'], 
             bins=30, alpha=0.7, color='#e74c3c', label='Poor Match')
    plt.title("TF-IDF: Score Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.hist(df[df['label']=='good_match']['lstm_score'], 
             bins=30, alpha=0.7, color='#2ecc71', label='Good Match')
    plt.hist(df[df['label']=='poor_match']['lstm_score'], 
             bins=30, alpha=0.7, color='#e74c3c', label='Poor Match')
    plt.title("LSTM: Score Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. ROC Curves
    plt.subplot(2, 3, 5)
    fpr_base, tpr_base, _ = roc_curve(y_true, df['tfidf_similarity'])
    fpr_lstm, tpr_lstm, _ = roc_curve(y_true, df['lstm_score'])
    
    auc_base = roc_auc_score(y_true, df['tfidf_similarity'])
    auc_lstm = roc_auc_score(y_true, df['lstm_score'])
    
    plt.plot(fpr_base, tpr_base, label=f'TF-IDF (AUC={auc_base:.3f})', linewidth=2, color='#3498db')
    plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC={auc_lstm:.3f})', linewidth=2, color='#9b59b6')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Box Plot Comparison
    plt.subplot(2, 3, 6)
    data_to_plot = [
        df[df['label']=='good_match']['tfidf_similarity'],
        df[df['label']=='poor_match']['tfidf_similarity'],
        df[df['label']=='good_match']['lstm_score'],
        df[df['label']=='poor_match']['lstm_score']
    ]
    positions = [1, 2, 4, 5]
    bp = plt.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                     labels=['TF-IDF\nGood', 'TF-IDF\nPoor', 'LSTM\nGood', 'LSTM\nPoor'])
    
    colors_box = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.ylabel("Score")
    plt.title("Score Distribution by Class", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_plot.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved to results/model_comparison_plot.png")

def plot_confusion_matrices(df):
    """Plot confusion matrices for both models"""
    y_true = df['label'].apply(lambda x: 1 if x == 'good_match' else 0).values
    
    # Calculate optimal thresholds
    fpr_base, tpr_base, thresh_base = roc_curve(y_true, df['tfidf_similarity'])
    optimal_idx_base = np.argmax(tpr_base - fpr_base)
    optimal_thresh_base = thresh_base[optimal_idx_base]
    
    fpr_lstm, tpr_lstm, thresh_lstm = roc_curve(y_true, df['lstm_score'])
    optimal_idx_lstm = np.argmax(tpr_lstm - fpr_lstm)
    optimal_thresh_lstm = thresh_lstm[optimal_idx_lstm]
    
    y_pred_base = (df['tfidf_similarity'] >= optimal_thresh_base).astype(int)
    y_pred_lstm = (df['lstm_score'] >= optimal_thresh_lstm).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_base = confusion_matrix(y_true, y_pred_base)
    cm_lstm = confusion_matrix(y_true, y_pred_lstm)
    
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Poor', 'Good'], yticklabels=['Poor', 'Good'])
    axes[0].set_title(f'TF-IDF Baseline\n(Threshold={optimal_thresh_base:.3f})', 
                      fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=['Poor', 'Good'], yticklabels=['Poor', 'Good'])
    axes[1].set_title(f'LSTM Model\n(Threshold={optimal_thresh_lstm:.3f})', 
                      fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("📊 Confusion matrices saved to results/confusion_matrices.png")

def run_evaluation():
    print("📊 Starting Comprehensive Model Evaluation...\n")
    
    df = load_predictions()
    if df is None:
        return

    y_true = df['label'].apply(lambda x: 1 if x == 'good_match' else 0).values
    
    # Calculate metrics for both models
    metrics_base = calculate_metrics(y_true, df['tfidf_similarity'], "TF-IDF Baseline")
    metrics_lstm = calculate_metrics(y_true, df['lstm_score'], "LSTM Siamese")
    
    results_df = pd.DataFrame([metrics_base, metrics_lstm])
    
    print("\n" + "="*70)
    print(" "*20 + "FINAL RESULTS COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    print("-" * 70)
    
    # Performance comparison
    lstm_auc = metrics_lstm['AUC']
    base_auc = metrics_base['AUC']
    auc_diff = lstm_auc - base_auc
    
    print("\n📈 Performance Analysis:")
    if auc_diff > 0.05:
        print(f"   LSTM significantly outperforms baseline (+{auc_diff:.4f} AUC)")
    elif auc_diff > 0:
        print(f"   LSTM marginally better (+{auc_diff:.4f} AUC)")
    elif auc_diff > -0.05:
        print(f"   Models perform similarly ({auc_diff:+.4f} AUC)")
    else:
        print(f"   Baseline outperforms LSTM ({auc_diff:+.4f} AUC)")
        print("   This suggests dataset quality issues or insufficient training data")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comparison(df)
    plot_confusion_matrices(df)
    
    # Save detailed results
    results_df.to_csv('results/model_metrics.csv', index=False)
    print("\nDetailed metrics saved to results/model_metrics.csv")
    
    print("\n" + "="*70)
    print("Evaluation complete! Check the results folder for visualizations.")
    print("="*70)

if __name__ == "__main__":
    run_evaluation()
