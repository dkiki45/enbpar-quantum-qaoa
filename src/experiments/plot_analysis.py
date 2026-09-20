import json
import matplotlib.pyplot as plt
from pathlib import Path

def get_metrics(folder_prefix, p_depth, num_seeds):
    energies = []
    probabilities = []
    
    for s in range(1, num_seeds + 1):
        path = Path(f"src/results/{folder_prefix}/p{p_depth}_seed{s}/summary.json")
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                energies.append(data["expectation_qubo"])
                probabilities.append(data["best"]["probability"])
    
    return energies, probabilities

if __name__ == "__main__":
    p = 3
    seeds = 15
    
    print("Loading datasets...")
    
    # Extract data for all 4 scenarios
    c_stress_e, c_stress_p = get_metrics("stress_n15_COBYLA", p, seeds)
    c_warm_e, c_warm_p = get_metrics("warm_start_n15_COBYLA", p, seeds)
    s_stress_e, s_stress_p = get_metrics("stress_n15_SPSA", p, seeds)
    s_warm_e, s_warm_p = get_metrics("warm_start_n15_SPSA", p, seeds)
    
    # Safety check
    if not s_warm_e:
        print("Warning: SPSA Warm-Start data not found yet. Please wait for the terminal to finish!")
        exit()

    out_dir = Path("src/results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        "COBYLA\n(Blind)", "COBYLA\n(Warm-Start)", 
        "SPSA\n(Blind)", "SPSA\n(Warm-Start)"
    ]
    colors = ['#ff9999', '#99ff99', '#66b3ff', '#c2c2f0']

    # ==========================================
    # PLOT 1: Master Energy Comparison
    # ==========================================
    plt.figure(figsize=(10, 6))
    box1 = plt.boxplot([c_stress_e, c_warm_e, s_stress_e, s_warm_e], 
                       labels=labels, patch_artist=True)
    
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title(f"Global Optimizer Performance (Depth p={p})\nEnergy Variance over {seeds} Seeds", fontsize=14)
    plt.ylabel("QUBO Energy (Lower is better)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(x=2.5, color='black', linestyle='-', alpha=0.8) # Separator line
    plt.savefig(out_dir / f"master_energy_comparison_p{p}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # PLOT 2: Master Probability Comparison
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    probs_pct = [
        [prob * 100 for prob in c_stress_p],
        [prob * 100 for prob in c_warm_p],
        [prob * 100 for prob in s_stress_p],
        [prob * 100 for prob in s_warm_p]
    ]
    
    box2 = plt.boxplot(probs_pct, labels=labels, patch_artist=True)
    
    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title(f"Measurement Probability of the Optimal Solution (p={p})\nSuccess Rate Validation", fontsize=14)
    plt.ylabel("Probability (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(x=2.5, color='black', linestyle='-', alpha=0.8) # Separator line
    plt.savefig(out_dir / f"master_probability_comparison_p{p}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nMaster analysis complete! All 4 quadrants successfully plotted.")
    print(f"Check the folder: {out_dir}")