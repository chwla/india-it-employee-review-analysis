import pandas as pd
import re
import os

def clean_text(text):
    """
    Standardize text: lowercase, remove special characters,
    normalize whitespace. Preserves technical symbols like +, #
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars but keep +, # for C++, C#, etc.
    text = re.sub(r'[^a-z0-9\+\#\.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_pipeline():
    raw_naukri = 'data/naukri_data_science_jobs_india.csv'
    output_dir = 'data/processed'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Output directory ready: {output_dir}")
    
    # --- Process Naukri Dataset ---
    if os.path.exists(raw_naukri):
        print(f"\n📊 Processing Naukri Data: {raw_naukri}...")
        try:
            df_naukri = pd.read_csv(raw_naukri)
            
            # Combine relevant columns
            df_naukri['full_text'] = (
                df_naukri['Job_Role'].fillna('') + " " + 
                df_naukri['Skills/Description'].fillna('')
            )
            
            # Clean text
            df_naukri['clean_text'] = df_naukri['full_text'].apply(clean_text)
            
            # Remove empty or very short entries
            df_naukri = df_naukri[df_naukri['clean_text'].str.len() > 20]
            
            # Save processed data
            out_file = os.path.join(output_dir, 'naukri_processed.csv')
            df_naukri[['clean_text']].to_csv(out_file, index=False)
            
            print(f"✅ Saved: {out_file}")
            print(f"   Rows: {len(df_naukri)}")
            print(f"   Avg length: {df_naukri['clean_text'].str.len().mean():.0f} chars")
            
        except Exception as e:
            print(f"❌ Error processing Naukri data: {e}")
    else:
        print(f"❌ Warning: File not found - {raw_naukri}")
        print("   Please ensure the Naukri dataset is in the data/ folder")
    
    print("\n" + "="*50)
    print("✅ Preprocessing Complete!")
    print("="*50)
    print("\nNext steps:")
    print("  1. Run: python build_dataset.py")
    print("  2. Run: python baseline_tfidf.py")
    print("  3. Run: python train_lstm.py")
    print("  4. Run: python evaluate.py")
    print("  5. Run: streamlit run dashboard.py")

if __name__ == "__main__":
    run_pipeline()