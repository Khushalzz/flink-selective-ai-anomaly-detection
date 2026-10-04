import json
import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import onnxruntime as ort

FEATURE_COLS = [
    "temperature", "humidity", "light", "voltage",
    "delta_temp_3m", "delta_volt_3m",
    "z_temp_15m", "z_volt_15m",
    "temp_slope_15m", "var_temp_15m"
]

class SensorAutoencoder(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 3), # Latent bottleneck
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("models", exist_ok=True)
    
    print("Loading train & validation features for Autoencoder...")
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    
    # Feature standardization (mean and std saved for ONNX pipeline)
    X_train_raw = train_df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    X_val_raw = val_df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.int32)
    
    means = X_train_raw.mean(axis=0)
    stds = X_train_raw.std(axis=0) + 1e-4
    
    X_train = (X_train_raw - means) / stds
    X_val = (X_val_raw - means) / stds
    
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = SensorAutoencoder(input_dim=len(FEATURE_COLS))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Training PyTorch Autoencoder (10 epochs)...")
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        avg_loss = total_loss / len(train_tensor)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:2d}/10 | Train MSE Loss: {avg_loss:.6f}")
            
    model.eval()
    with torch.no_grad():
        val_recon = model(val_tensor)
        # Reconstruction error per sample (MSE across the 10 features)
        val_mse = torch.mean((val_tensor - val_recon) ** 2, dim=1).numpy()
        
    print(f"Validation Reconstruction Error: Mean={val_mse.mean():.4f}, 95th-pctile={np.percentile(val_mse, 95):.4f}")
    
    # Calibrate MSE reconstruction error to P(anomaly) via LogisticRegression
    calibrator = LogisticRegression(class_weight="balanced")
    calibrator.fit(val_mse.reshape(-1, 1), y_val)
    
    calib_params = {
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "features": FEATURE_COLS
    }
    with open("models/ae_calibration.json", "w") as f:
        json.dump(calib_params, f, indent=2)
    print(f"Calibrated probability model saved: P(anomaly) = sigmoid({calib_params['coef']:.4f} * MSE + {calib_params['intercept']:.4f})")
    
    # Export Autoencoder to ONNX
    onnx_path = "models/autoencoder.onnx"
    dummy_input = torch.randn(1, len(FEATURE_COLS), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["float_input"],
        output_names=["reconstructed"],
        dynamic_axes={"float_input": {0: "batch_size"}, "reconstructed": {0: "batch_size"}}
    )
    print(f"Exported PyTorch Autoencoder to ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    
    # Verify ONNX model execution
    session = ort.InferenceSession(onnx_path)
    test_recon = session.run(None, {"float_input": X_val[:5]})[0]
    print(f"ONNX Test Inference Succeeded! Output shape: {test_recon.shape}")

if __name__ == "__main__":
    main()
