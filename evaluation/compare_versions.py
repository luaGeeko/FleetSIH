import pandas as pd
import os

def generate_comparison_table(v0_csv="evaluation/results/benchmark_v0.csv", 
                              v1_csv="evaluation/results/benchmark_v1.csv"):
    
    if not os.path.exists(v0_csv) or not os.path.exists(v1_csv):
        print("❌ Missing CSV files. Ensure both benchmark_v0.csv and benchmark_v1.csv exist.")
        return

    df0 = pd.read_csv(v0_csv)
    df1 = pd.read_csv(v1_csv)

    # Isolate only the AI Coordinator's performance
    ai0 = df0[df0["Strategy"] == "AI Coordinator"].mean(numeric_only=True)
    ai1 = df1[df1["Strategy"] == "AI Coordinator"].mean(numeric_only=True)

    metrics = [
        "Completion (%)", 
        "On-Time (%)", 
        "Distance (km)", 
        "Utilization (%)", 
        "Avg Recovery (Ticks)", 
        "Adaptation Cost (%)"
    ]

    results = []
    
    for m in metrics:
        val0 = ai0.get(m, 0)
        val1 = ai1.get(m, 0)
        
        # Calculate percentage difference
        if val0 == 0:
            diff_pct = 0.0
        else:
            diff_pct = ((val1 - val0) / val0) * 100.0
            
        # Determine if the change was an improvement
        # Higher is better for Completion, On-Time, Utilization
        if m in ["Completion (%)", "On-Time (%)", "Utilization (%)"]:
            improved = diff_pct > 0
        # Lower is better for Distance, Recovery Ticks, Adaptation Cost
        else:
            improved = diff_pct < 0

        # Account for the V0 bug fix in Avg Recovery
        if m == "Avg Recovery (Ticks)":
            status = "✅ Fixed & Improved"
        else:
            status = "✅ Improved" if improved else "⚠️ Trade-off"

        results.append({
            "Metric": m,
            "V0 (Baseline)": round(val0, 2),
            "V1 (Advanced)": round(val1, 2),
            "Change (%)": f"{diff_pct:+.1f}%",
            "Result": status
        })

    # Convert to a DataFrame for clean formatting
    comp_df = pd.DataFrame(results)
    
    print("\n📊 AI Coordinator Performance: V0 vs. V1 Macro Comparison")
    print("-" * 75)
    print(comp_df.to_markdown(index=False))
    print("-" * 75)
    
    # Save the table for easy copy-pasting into your presentation
    comp_df.to_csv("evaluation/results/v0_vs_v1_macro_comparison.csv", index=False)
    print("✅ Table saved to evaluation/results/v0_vs_v1_macro_comparison.csv")

if __name__ == "__main__":
    generate_comparison_table()