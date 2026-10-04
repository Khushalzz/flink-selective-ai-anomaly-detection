import os
import numpy as np
import pandas as pd

def compute_dual_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes dual-window time-series features per sensor mote:
    - Short window (approx 3 min / 6 readings) -> instantaneous shock
    - Long window (approx 15 min / 30 readings) -> baseline mean, std, z-score, trend slope
    """
    print("Computing dual-window features per mote...")
    features_list = []
    
    # Process mote by mote to maintain clean state
    for mote, mdf in df.groupby("moteid", sort=False):
        mdf = mdf.copy()
        
        # 3-minute short window features (approx 6 readings @ 31s sample rate)
        mdf["short_temp_mean"] = mdf["temperature"].rolling(window=6, min_periods=1).mean()
        mdf["short_volt_mean"] = mdf["voltage"].rolling(window=6, min_periods=1).mean()
        mdf["delta_temp_3m"] = mdf["temperature"] - mdf["short_temp_mean"]
        mdf["delta_volt_3m"] = mdf["voltage"] - mdf["short_volt_mean"]
        
        # 15-minute long window features (approx 30 readings)
        mdf["long_temp_mean"] = mdf["temperature"].rolling(window=30, min_periods=3).mean()
        mdf["long_temp_std"] = mdf["temperature"].rolling(window=30, min_periods=3).std().fillna(0.1)
        mdf["long_volt_mean"] = mdf["voltage"].rolling(window=30, min_periods=3).mean()
        mdf["long_volt_std"] = mdf["voltage"].rolling(window=30, min_periods=3).std().fillna(0.01)
        
        # Z-scores
        mdf["z_temp_15m"] = (mdf["temperature"] - mdf["long_temp_mean"]) / (mdf["long_temp_std"] + 1e-4)
        mdf["z_volt_15m"] = (mdf["voltage"] - mdf["long_volt_mean"]) / (mdf["long_volt_std"] + 1e-4)
        
        # Slope over 15-min window
        mdf["temp_slope_15m"] = (mdf["temperature"] - mdf["temperature"].shift(10)).fillna(0.0) / 10.0
        mdf["var_temp_15m"] = (mdf["long_temp_std"] ** 2).fillna(0.0)
        
        features_list.append(mdf)
        
    res_df = pd.concat(features_list).sort_index()
    return res_df

def inject_controlled_anomalies(df: pd.DataFrame, target_prevalence: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """
    Injects 3 realistic anomaly classes following TimeEval / TSB-UAD protocols:
    1. Point Spikes (1.2%): extreme impulse (+/- 4 to 8 sigma)
    2. Contextual Drift (1.0%): gradual degradation / slope shift over 30 mins
    3. Flatlines / Dropout (0.8%): stuck sensor values
    """
    np.random.seed(seed)
    df = df.copy()
    n = len(df)
    df["label"] = 0
    df["anomaly_type"] = "NORMAL"
    
    # 1. Point Spikes (approx 1.2% of data)
    n_spikes = int(n * 0.012)
    spike_idx = np.random.choice(n, size=n_spikes, replace=False)
    for idx in spike_idx:
        # Either temp spike or voltage drop
        if np.random.rand() > 0.5:
            # Temperature spike
            shock = np.random.uniform(8.0, 20.0) * (1 if np.random.rand() > 0.3 else -1)
            df.iloc[idx, df.columns.get_loc("temperature")] += shock
            df.iloc[idx, df.columns.get_loc("anomaly_type")] = "SPIKE_TEMP"
        else:
            # Voltage drop
            drop = np.random.uniform(0.4, 0.9)
            df.iloc[idx, df.columns.get_loc("voltage")] = max(1.8, df.iloc[idx]["voltage"] - drop)
            df.iloc[idx, df.columns.get_loc("anomaly_type")] = "SPIKE_VOLT_DROP"
        df.iloc[idx, df.columns.get_loc("label")] = 1

    # 2. Contextual Drift (approx 1.0% of data in blocks of 30-50 readings)
    n_drift_blocks = int((n * 0.010) / 40)
    for _ in range(n_drift_blocks):
        start = np.random.randint(100, n - 60)
        length = np.random.randint(25, 45)
        drift_slope = np.random.uniform(0.2, 0.5)
        for step in range(length):
            cur = start + step
            df.iloc[cur, df.columns.get_loc("temperature")] += (step * drift_slope)
            df.iloc[cur, df.columns.get_loc("label")] = 1
            df.iloc[cur, df.columns.get_loc("anomaly_type")] = "DRIFT"

    # 3. Flatline / Stuck Sensor (approx 0.8% of data in blocks of 20-30 readings)
    n_flat_blocks = int((n * 0.008) / 25)
    for _ in range(n_flat_blocks):
        start = np.random.randint(100, n - 40)
        length = np.random.randint(15, 30)
        stuck_temp = df.iloc[start]["temperature"]
        stuck_volt = df.iloc[start]["voltage"]
        for step in range(length):
            cur = start + step
            df.iloc[cur, df.columns.get_loc("temperature")] = stuck_temp
            df.iloc[cur, df.columns.get_loc("voltage")] = stuck_volt
            df.iloc[cur, df.columns.get_loc("anomaly_type")] = "FLATLINE"
            df.iloc[cur, df.columns.get_loc("label")] = 1

    actual_prevalence = df["label"].mean()
    print(f"Injected anomalies complete! Actual prevalence: {actual_prevalence*100:.2f}% ({df['label'].sum():,} anomalies)")
    print(df["anomaly_type"].value_counts())
    return df

def main():
    os.makedirs("data/processed", exist_ok=True)
    parquet_path = "data_chronological.parquet"
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return

    print("Loading chronological sensor data...")
    # Load first 200,000 rows for high-fidelity training & evaluation benchmark
    df = pd.read_parquet(parquet_path).head(200000)
    print(f"Loaded {len(df):,} events.")

    # Split into Train (first 60%), Validation (20%), and Test (20%)
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    train_df = df.iloc[:train_end].copy()
    val_raw = df.iloc[train_end:val_end].copy()
    test_raw = df.iloc[val_end:].copy()

    # Train set stays clean (normal readings only) for unsupervised IF / Autoencoder training
    train_df["label"] = 0
    train_df["anomaly_type"] = "NORMAL"
    train_feat = compute_dual_window_features(train_df)

    # Validation and Test sets get controlled anomaly injection
    print("\n--- Injecting anomalies into Validation Set ---")
    val_injected = inject_controlled_anomalies(val_raw)
    val_feat = compute_dual_window_features(val_injected)

    print("\n--- Injecting anomalies into Test Set ---")
    test_injected = inject_controlled_anomalies(test_raw)
    test_feat = compute_dual_window_features(test_injected)

    train_feat.to_parquet("data/processed/train_features.parquet", index=False)
    val_feat.to_parquet("data/processed/val_features.parquet", index=False)
    test_feat.to_parquet("data/processed/test_features.parquet", index=False)

    print("\nSaved:")
    print(f"- data/processed/train_features.parquet: {len(train_feat):,} rows (Clean normal)")
    print(f"- data/processed/val_features.parquet:   {len(val_feat):,} rows (With injection)")
    print(f"- data/processed/test_features.parquet:  {len(test_feat):,} rows (With injection)")

if __name__ == "__main__":
    main()
