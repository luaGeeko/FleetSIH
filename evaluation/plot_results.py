import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_hackathon_plots(csv_path="evaluation/results/latest_benchmark.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ Could not find {csv_path}. Run benchmark.py first!")
        return

    df = pd.read_csv(csv_path)
    
    sns.set_theme(style="whitegrid", palette="muted")
    # Expanding to a 4x2 grid for the new Adaptation Cost metric
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle("Smart Fleet Platform: Controlled Experiment & Adaptation Cost", fontsize=22, fontweight='bold', y=0.98)

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
    
    # Clear the unused 8th subplot
    axes[3, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "evaluation/results/performance_dashboard.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plots successfully generated and saved to {save_path}")

if __name__ == "__main__":
    generate_hackathon_plots()