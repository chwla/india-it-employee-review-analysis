"""
Sprint 0 Smoke Test Script
Validates that sample data loads correctly and performs basic sanity checks
"""

import pandas as pd
import os
import sys

def check_file_exists(filepath):
    """Check if a file exists"""
    if not os.path.exists(filepath):
        print(f"❌ FAIL: File not found - {filepath}")
        return False
    print(f"✅ PASS: File found - {filepath}")
    return True

def load_and_validate_data():
    """Load sample data and perform sanity checks"""
    
    print("=" * 60)
    print("SPRINT 0 SMOKE TEST - Resume-JD Matching Project")
    print("=" * 60)
    
    # Check if sample data exists
    sample_file = "data/samples/labeled_pairs.csv"
    if not check_file_exists(sample_file):
        print("\n❌ Sample data not found. Run create_sample_data.py first.")
        return False
    
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    try:
        # Load labeled pairs
        df = pd.read_csv(sample_file)
        print(f"✅ Successfully loaded {len(df)} labeled pairs")
        
        # Basic statistics
        print("\n" + "-" * 60)
        print("DATA STATISTICS")
        print("-" * 60)
        print(f"Total pairs: {len(df)}")
        print(f"Good matches: {len(df[df['label'] == 'good_match'])}")
        print(f"Poor matches: {len(df[df['label'] == 'poor_match'])}")
        
        # Check label balance
        good_count = len(df[df['label'] == 'good_match'])
        poor_count = len(df[df['label'] == 'poor_match'])
        balance_ratio = min(good_count, poor_count) / max(good_count, poor_count)
        
        print(f"\nLabel balance ratio: {balance_ratio:.2f}")
        if balance_ratio < 0.3:
            print("⚠️  WARNING: Dataset is highly imbalanced")
        else:
            print("✅ Dataset balance is acceptable")
        
        # Check for empty fields
        print("\n" + "-" * 60)
        print("DATA QUALITY CHECKS")
        print("-" * 60)
        
        empty_resume = df['resume_text'].isna().sum() or (df['resume_text'] == '').sum()
        empty_jd = df['jd_text'].isna().sum() or (df['jd_text'] == '').sum()
        
        if empty_resume > 0:
            print(f"❌ FAIL: Found {empty_resume} empty resume texts")
            return False
        print("✅ PASS: No empty resume texts")
        
        if empty_jd > 0:
            print(f"❌ FAIL: Found {empty_jd} empty JD texts")
            return False
        print("✅ PASS: No empty JD texts")
        
        # Sample one resume and one JD
        print("\n" + "=" * 60)
        print("SAMPLE DOCUMENT ANALYSIS")
        print("=" * 60)
        
        sample_row = df.iloc[0]
        resume_text = sample_row['resume_text']
        jd_text = sample_row['jd_text']
        
        print("\n📄 Sample Resume (ID: " + sample_row['resume_id'] + "):")
        print("-" * 60)
        print(resume_text[:200] + "..." if len(resume_text) > 200 else resume_text)
        
        print("\n📋 Sample Job Description (ID: " + sample_row['jd_id'] + "):")
        print("-" * 60)
        print(jd_text[:200] + "..." if len(jd_text) > 200 else jd_text)
        
        # Text length analysis
        print("\n" + "-" * 60)
        print("TEXT LENGTH STATISTICS")
        print("-" * 60)
        
        resume_lengths = df['resume_text'].str.len()
        jd_lengths = df['jd_text'].str.len()
        
        print(f"Resume text - Min: {resume_lengths.min()}, Max: {resume_lengths.max()}, Mean: {resume_lengths.mean():.1f}")
        print(f"JD text     - Min: {jd_lengths.min()}, Max: {jd_lengths.max()}, Mean: {jd_lengths.mean():.1f}")
        
        # Token count estimation (rough approximation: chars / 4)
        resume_tokens = resume_lengths / 4
        jd_tokens = jd_lengths / 4
        
        print(f"\nEstimated token counts:")
        print(f"Resume - Min: {resume_tokens.min():.0f}, Max: {resume_tokens.max():.0f}, Mean: {resume_tokens.mean():.0f}")
        print(f"JD     - Min: {jd_tokens.min():.0f}, Max: {jd_tokens.max():.0f}, Mean: {jd_tokens.mean():.0f}")
        
        # Check if texts are within reasonable token limits (512 is our target)
        max_resume_tokens = resume_tokens.max()
        max_jd_tokens = jd_tokens.max()
        
        if max_resume_tokens > 512 or max_jd_tokens > 512:
            print(f"\n⚠️  WARNING: Some documents may exceed 512 token limit")
            print(f"   Consider truncation in preprocessing")
        else:
            print(f"\n✅ All documents within 512 token limit")
        
        # Word count analysis
        print("\n" + "-" * 60)
        print("WORD COUNT STATISTICS")
        print("-" * 60)
        
        resume_words = df['resume_text'].str.split().str.len()
        jd_words = df['jd_text'].str.split().str.len()
        
        print(f"Resume - Min: {resume_words.min()}, Max: {resume_words.max()}, Mean: {resume_words.mean():.1f}")
        print(f"JD     - Min: {jd_words.min()}, Max: {jd_words.max()}, Mean: {jd_words.mean():.1f}")
        
        # Final validation
        print("\n" + "=" * 60)
        print("FINAL VALIDATION")
        print("=" * 60)
        
        checks_passed = True
        
        # Check 1: Sufficient data
        if len(df) >= 20:
            print(f"✅ PASS: Have {len(df)} labeled pairs (minimum 20)")
        else:
            print(f"❌ FAIL: Only {len(df)} pairs (need minimum 20)")
            checks_passed = False
        
        # Check 2: Label distribution
        if good_count > 0 and poor_count > 0:
            print(f"✅ PASS: Both labels present (good: {good_count}, poor: {poor_count})")
        else:
            print(f"❌ FAIL: Missing label class")
            checks_passed = False
        
        # Check 3: No missing data
        if df.isna().sum().sum() == 0:
            print("✅ PASS: No missing values in dataset")
        else:
            print(f"⚠️  WARNING: Found {df.isna().sum().sum()} missing values")
        
        # Check 4: Unique pairs
        if len(df[['resume_id', 'jd_id']].drop_duplicates()) == len(df):
            print("✅ PASS: All resume-JD pairs are unique")
        else:
            print("⚠️  WARNING: Found duplicate pairs")
        
        print("\n" + "=" * 60)
        if checks_passed:
            print("🎉 SPRINT 0 SMOKE TEST PASSED!")
            print("=" * 60)
            print("Next steps:")
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Proceed to Sprint 1: Data pipeline & labeling")
            return True
        else:
            print("❌ SPRINT 0 SMOKE TEST FAILED")
            print("=" * 60)
            print("Please fix the issues above before proceeding.")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_and_validate_data()
    sys.exit(0 if success else 1)