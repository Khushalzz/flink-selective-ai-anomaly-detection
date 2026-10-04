import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs("experiments/results", exist_ok=True)
    
    # Load 6-system results
    results_path = "experiments/results/rigorous_6_system_results.csv"
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found. Run evaluate_systems.py first.")
        return
        
    df = pd.read_csv(results_path)
    
    # Set aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    system_f_name = str(df["System"].iloc[5]).lower()
    system_f_is_legacy = "legacy" in system_f_name and "proxy" in system_f_name
    if system_f_is_legacy:
        fig.suptitle("Offline test-split results; F is an unverified legacy proxy", fontsize=14)
    system_f_label = "F: legacy proxy" if system_f_is_legacy else "F: +Laya"
    system_labels = [
        "A: IF",
        "B: AE",
        "C: IF+AE",
        "D: +XGBoost",
        "E: +Heuristic",
        system_f_label,
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#E377C2", "#17BECF"]
    x = np.arange(len(system_labels))
    width = 0.26
    
    # -------------------------------------------------------------
    # 1. Detection Quality: F1, PR-AUC, ROC-AUC
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    b1 = ax1.bar(x - width, df["F1"], width, label="F1-Score", color="#1f77b4", alpha=0.9)
    b2 = ax1.bar(x, df["PR-AUC"], width, label="PR-AUC", color="#ff7f0e", alpha=0.9)
    b3 = ax1.bar(x + width, df["ROC-AUC"], width, label="ROC-AUC", color="#2ca02c", alpha=0.9)
    
    ax1.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax1.set_title("Overall Anomaly Detection Quality (Test Split)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    
    for i in range(len(system_labels)):
        ax1.text(i - width, df["F1"].iloc[i] + 0.02, f"{df['F1'].iloc[i]:.2f}", ha='center', fontsize=8, rotation=45)
        ax1.text(i, df["PR-AUC"].iloc[i] + 0.02, f"{df['PR-AUC'].iloc[i]:.2f}", ha='center', fontsize=8, rotation=45)
        ax1.text(i + width, df["ROC-AUC"].iloc[i] + 0.02, f"{df['ROC-AUC'].iloc[i]:.2f}", ha='center', fontsize=8, rotation=45)
        
    # -------------------------------------------------------------
    # 2. False Alarm Rate (FPR %) & Precision
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    p1 = ax2.bar(x - 0.18, df["FPR"] * 100.0, width=0.35, color="#d62728", alpha=0.85, label="FPR % (Lower is Better)")
    p2 = ax2_twin.bar(x + 0.18, df["Precision"], width=0.35, color="#9467bd", alpha=0.85, label="Precision (Higher is Better)")
    
    ax2.set_ylabel("False Positive Rate (%)", color="#d62728", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("Precision (PPV)", color="#9467bd", fontsize=11, fontweight="bold")
    ax2.set_title("False Alarms vs. Anomaly Precision", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(system_labels, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(df["FPR"] * 100.0) * 1.3)
    ax2_twin.set_ylim(0, 1.1)
    
    for i in range(len(system_labels)):
        fpr_val = df["FPR"].iloc[i] * 100.0
        prec_val = df["Precision"].iloc[i]
        ax2.text(i - 0.18, fpr_val + 0.3, f"{fpr_val:.1f}%", ha='center', fontsize=8, color="#900")
        ax2_twin.text(i + 0.18, prec_val + 0.02, f"{prec_val:.2f}", ha='center', fontsize=8, color="#400080")
        
    # -------------------------------------------------------------
    # 3. Precision vs Recall Trade-off Frontier
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df["Recall"], df["Precision"], s=220, c=colors, edgecolors="black", linewidth=1.5, zorder=5)
    for i, txt in enumerate(system_labels):
        offset_y = 0.02 if i % 2 == 0 else -0.03
        ax3.annotate(f" {txt}", (df["Recall"].iloc[i], df["Precision"].iloc[i] + offset_y),
                     fontsize=9, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.2, edgecolor=colors[i]))
        
    ax3.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Precision (PPV)", fontsize=11, fontweight="bold")
    ax3.set_title("Precision vs. Recall Trade-off Frontier", fontsize=12, fontweight="bold")
    ax3.set_xlim(0.60, 1.02)
    ax3.set_ylim(0.05, 0.95)
    ax3.grid(True, linestyle="--", alpha=0.6)
    
    # -------------------------------------------------------------
    # 4. Disaggregated Recall by Anomaly Family
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    
    # Parse percentages from strings if necessary
    def parse_pct(val):
        if isinstance(val, str):
            return float(val.replace("%", "").strip())
        return float(val) * 100.0
        
    spikes = [parse_pct(v) for v in df["Spikes (Temp/Volt)"]]
    drifts = [parse_pct(v) for v in df["Contextual Drift"]]
    flatlines = [parse_pct(v) for v in df["Sensor Flatline"]]
    
    w4 = 0.25
    ax4.bar(x - w4, spikes, w4, label="Point Spikes", color="#e74c3c", alpha=0.9)
    ax4.bar(x, drifts, w4, label="Contextual Drifts", color="#f39c12", alpha=0.9)
    ax4.bar(x + w4, flatlines, w4, label="Sensor Flatlines", color="#2ecc71", alpha=0.9)
    
    ax4.set_ylabel("Recall (%)", fontsize=11, fontweight="bold")
    ax4.set_title("Disaggregated Recall by Anomaly Family", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(system_labels, fontsize=10, fontweight="bold")
    ax4.set_ylim(0, 115)
    ax4.legend(loc="lower right", frameon=True)
    ax4.grid(axis="y", linestyle="--", alpha=0.6)
    
    for i in range(len(system_labels)):
        ax4.text(i - w4, spikes[i] + 1.5, f"{spikes[i]:.0f}%", ha='center', fontsize=7.5)
        ax4.text(i, drifts[i] + 1.5, f"{drifts[i]:.0f}%", ha='center', fontsize=7.5)
        ax4.text(i + w4, flatlines[i] + 1.5, f"{flatlines[i]:.0f}%", ha='center', fontsize=7.5)
        
    plt.tight_layout()
    result_chart = "experiments/results/rigorous_6_system_comparison.png"
    asset_chart = "assets/rigorous_6_system_comparison.png"
    os.makedirs("assets", exist_ok=True)
    fig.savefig(result_chart, dpi=300)
    fig.savefig(asset_chart, dpi=300)
    print(f"Saved offline comparison figures to {result_chart} and {asset_chart}")

if __name__ == "__main__":
    main()
