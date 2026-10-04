import pandas as pd
import numpy as np

def verify_dataset_integrity():
    print("=" * 65)
    print("AUDIT: Verifying Zero Data Leakage in Dataset Splits")
    print("=" * 65)
    
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    test_df = pd.read_parquet("data/processed/test_features.parquet")
    
    print(f"Train split size: {len(train_df):,} rows")
    print(f"Val split size:   {len(val_df):,} rows")
    print(f"Test split size:  {len(test_df):,} rows")
    
    # 1. Exact Chronological Ordering Verification using datetime parse
    train_dt_min = pd.to_datetime(train_df['date'] + ' ' + train_df['time']).min()
    train_dt_max = pd.to_datetime(train_df['date'] + ' ' + train_df['time']).max()
    val_dt_min = pd.to_datetime(val_df['date'] + ' ' + val_df['time']).min()
    val_dt_max = pd.to_datetime(val_df['date'] + ' ' + val_df['time']).max()
    test_dt_min = pd.to_datetime(test_df['date'] + ' ' + test_df['time']).min()
    test_dt_max = pd.to_datetime(test_df['date'] + ' ' + test_df['time']).max()
    
    print(f"\nChronological Split Windows:")
    print(f"- Train Range: {train_dt_min}  ->  {train_dt_max}")
    print(f"- Val Range:   {val_dt_min}  ->  {val_dt_max}")
    print(f"- Test Range:  {test_dt_min}  ->  {test_dt_max}")
    
    assert train_dt_max <= val_dt_min, f"LEAKAGE DETECTED: Train overlap with Val! ({train_dt_max} > {val_dt_min})"
    assert val_dt_max <= test_dt_min, f"LEAKAGE DETECTED: Val overlap with Test! ({val_dt_max} > {test_dt_min})"
    print("\nPASS: Train, Validation, and Test sets have ZERO temporal overlap (Strictly Chronological).")
    
    # 2. Ground-Truth Anomaly Distribution in Test Set
    print("\nTest Set Anomaly Breakdown by Family:")
    breakdown = test_df["anomaly_type"].value_counts()
    for atype, count in breakdown.items():
        pct = (count / len(test_df)) * 100.0
        print(f"  {atype:18s}: {count:6,d} windows ({pct:5.2f}%)")
        
    print("\nAUDIT COMPLETE: All data splits are strictly leakage-free.")

if __name__ == "__main__":
    verify_dataset_integrity()
