import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def verify_and_plot_model_meta(df, version):
    """Verifies correct model loading and generates an independent metadata plot."""
    ai_data = df[df["Strategy"] == "AI Coordinator"]
    
    if ai_data.empty or "Model Used" not in df.columns or "Observation Shape" not in df.columns:
        print("⚠️ Model Used/Observation Shape columns missing or no AI data found.")
        return

    unique_models = ai_data["Model Used"].unique()
    unique_shapes = ai_data["Observation Shape"].unique()

    print(f"\n🔍 VERIFICATION CHECK [{version.upper()}]:")
    print(f"   - Model Path(s): {', '.join(map(str, unique_models))}")
    print(f"   - Obs Shape(s): {', '.join(map(str, unique_shapes))}")

    # Strict MTech-level validation warnings
    if version == "v0" and any(str(s) == "84" for s in unique_shapes):
        print("   ❌ WARNING: Found 84-dim (V1) data in your V0 benchmark data!")
    elif version == "v1" and any(str(s) == "72" for s in unique_shapes):
        print("   ❌ WARNING: Found 72-dim (V0) data in your V1 benchmark data!")
    else:
        print("   ✅ Configuration matches expected version parameters.")

    # Create the independent metadata plot
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')
    
    model_str = str(unique_models[0]) if len(unique_models) > 0 else "N/A"
    shape_str = f"{unique_shapes[0]} dimensions" if len(unique_shapes) > 0 else "N/A"
    
    table_data = [
        ["Attribute", "Configuration Validated"],
        ["Platform Version", version.upper()],
        ["Observation Space", shape_str],
        ["Model Checkpoint", model_str]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.35, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table for a presentation
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4C72B0') # Seaborn muted blue
        else:
            cell.set_facecolor('#F8F9FA' if row % 2 == 0 else '#FFFFFF')

    plt.title(f"AI Coordinator Architecture [{version.upper()}]", fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    
    meta_save_path = f"evaluation/results/model_metadata_{version}.png"
    plt.savefig(meta_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Architecture metadata plot saved to {meta_save_path}\n")


def generate_hackathon_plots(csv_path="evaluation/results/benchmark_v1.csv", version="v1"):
    if not os.path.exists(csv_path):
        print(f"❌ Could not find {csv_path}. Run benchmark.py first!")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Run Verification & Generate Independent Meta Plot
    verify_and_plot_model_meta(df, version)
    
    # 2. Generate Primary Dashboard
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(f"Smart Fleet Platform [{version.upper()}]: Controlled Experiment Results", fontsize=22, fontweight='bold', y=0.98)

    # --- 1. Completion Rate ---
    sns.barplot(data=df, x="Scenario", y="Completion (%)", hue="Strategy", ax=axes[0, 0])
    axes[0, 0].set_title("Mission Success: Completion Rate", fontsize=14)
    axes[0, 0].set_ylabel("Completion (%) ↑")
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].legend(loc='lower right')

    # --- 2. On-Time Delivery ---
    sns.barplot(data=df, x="Scenario", y="On-Time (%)", hue="Strategy", ax=axes[0, 1])
    axes[0, 1].set_title("Reliability: On-Time Delivery", fontsize=14)
    axes[0, 1].set_ylabel("On-Time (%) ↑")
    axes[0, 1].set_ylim(0, 110)
    axes[0, 1].legend(loc='lower right')

    # --- 3. Scalability (Distance by Fleet Size) ---
    sns.lineplot(
        data=df, x="Fleet Size", y="Distance (km)", hue="Strategy", 
        marker="o", linewidth=3, markersize=10, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Static Efficiency: Total Distance vs. Fleet Scalability", fontsize=14)
    axes[1, 0].set_ylabel("Distance (km) ↓")
    axes[1, 0].set_xticks(df["Fleet Size"].unique())
    axes[1, 0].legend(loc='upper left')

    # --- 4. Fleet Utilization ---
    util_df = df.groupby(["Strategy", "Seed"])["Utilization (%)"].mean().reset_index()
    sns.barplot(data=util_df, x="Strategy", y="Utilization (%)", ax=axes[1, 1])
    axes[1, 1].set_title("Resource Management: Average Fleet Utilization", fontsize=14)
    axes[1, 1].set_ylabel("Utilization (%) ↑")
    axes[1, 1].set_ylim(0, 100)

    # --- 5. Disruption Recovery Rate ---
    disruption_df = df[df["Scenario"] != "Normal Day"]
    sns.barplot(data=disruption_df, x="Scenario", y="Recovery Rate (%)", hue="Strategy", ax=axes[2, 0])
    axes[2, 0].set_title("Resilience: Disruption Recovery Rate", fontsize=14)
    axes[2, 0].set_ylabel("Recovery Rate (%) ↑")
    axes[2, 0].set_ylim(0, 110)
    axes[2, 0].legend(loc='lower right')

    # --- 6. Recovery Latency (Ticks) ---
    sns.barplot(data=disruption_df, x="Scenario", y="Avg Recovery (Ticks)", hue="Strategy", ax=axes[2, 1])
    axes[2, 1].set_title("Agility: Average Ticks to Recover", fontsize=14)
    axes[2, 1].set_ylabel("Recovery Time (Ticks) ↓")
    axes[2, 1].legend(loc='upper right')
    
    # --- 7. Adaptation Cost (Distance Penalty) ---
    sns.barplot(data=disruption_df, x="Scenario", y="Adaptation Cost (%)", hue="Strategy", ax=axes[3, 0])
    axes[3, 0].set_title("Adaptation Cost: Extra Distance Caused by Disruption", fontsize=14)
    axes[3, 0].set_ylabel("Distance Penalty (%) ↓")
    axes[3, 0].legend(loc='upper right')
    
    # --- 8. NEW: Service-Level Constraint Cost ---
    if "Constraint Cost" in df.columns:
        sns.barplot(data=df, x="Scenario", y="Constraint Cost", hue="Strategy", ax=axes[3, 1])
        axes[3, 1].set_title("Service Reliability: Constraint Cost (Lateness & Unserved)", fontsize=14)
        axes[3, 1].set_ylabel("Cost Penalty ↓")
        axes[3, 1].legend(loc='upper right')
    else:
        axes[3, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save dynamically based on version
    save_path = f"evaluation/results/performance_dashboard_{version}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Dashboard successfully generated and saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Smart Fleet Platform Benchmarks")
    parser.add_argument(
        "--version", 
        type=str, 
        choices=["v0", "v1"], 
        default="v1", 
        help="Select which AI version to plot: 'v0' (Baseline) or 'v1' (Advanced Logistics)"
    )
    args = parser.parse_args()
    
    target_csv = f"evaluation/results/benchmark_{args.version}.csv"
    generate_hackathon_plots(csv_path=target_csv, version=args.version)