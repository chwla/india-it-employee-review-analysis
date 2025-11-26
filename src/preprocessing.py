import pandas as pd
import re
import os

def clean_text(text):

    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars (keep +, # for C++, C#)
    # Replacing non-alphanumeric (except +, #, .) with space
    text = re.sub(r'[^a-z0-9\+\#\.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_pipeline():
    # Define paths
    # Note: Naukri CSV is in 'data/' based on your ls output
    files = {
        'labeled': 'data/samples/labeled_pairs.csv',
        'naukri': 'data/naukri_data_science_jobs_india.csv'
    }
    output_dir = 'data/processed'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")
    
    # --- Process 1: Labeled Pairs (Training Data) ---
    if os.path.exists(files['labeled']):
        print(f"\nProcessing Labeled Data: {files['labeled']}...")
        df = pd.read_csv(files['labeled'])
        
        # Clean both text columns
        df['clean_resume'] = df['resume_text'].apply(clean_text)
        df['clean_jd'] = df['jd_text'].apply(clean_text)
        
        # Save
        out_file = os.path.join(output_dir, 'labeled_pairs_processed.csv')
        df.to_csv(out_file, index=False)
        print(f"Saved processed labeled data: {out_file} ({len(df)} rows)")
    else:
        print(f"Warning: File not found - {files['labeled']}")

    # --- Process 2: Naukri Dataset (Vocabulary/Pre-training) ---
    if os.path.exists(files['naukri']):
        print(f"\nProcessing Naukri Data: {files['naukri']}...")
        try:
            df_naukri = pd.read_csv(files['naukri'])
            
            # The useful text is mainly in 'Skills/Description' or 'Job_Role'
            # Let's combine them for a rich text representation
            df_naukri['full_text'] = df_naukri['Job_Role'] + " " + df_naukri['Skills/Description']
            
            # Clean
            df_naukri['clean_text'] = df_naukri['full_text'].apply(clean_text)
            
            # Save only the clean text (useful for training embeddings/vocab)
            out_file = os.path.join(output_dir, 'naukri_processed.csv')
            df_naukri[['clean_text']].to_csv(out_file, index=False)
            print(f"Saved processed Naukri data: {out_file} ({len(df_naukri)} rows)")
            
        except Exception as e:
            print(f"Error processing Naukri data: {e}")
    else:
        print(f"Warning: File not found - {files['naukri']}")

if __name__ == "__main__":
    run_pipeline()