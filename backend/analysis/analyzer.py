import json
import os

def analyze():
    # Make sure we're in the right directory or dynamically path to output
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    metrics_path = os.path.join(base_dir, "output", "metrics.json")
    
    if not os.path.exists(metrics_path):
        print("No metrics found. Run training first.")
        return
        
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    episodes = len(data)
    if episodes == 0:
        print("No complete episodes recorded.")
        return
        
    # The collapses are cumulative in the environment so final step has total
    total_collapses = data[-1].get("collapses", 0)
    
    initial_ideologies = data[0]["ideologies"]
    final_ideologies = data[-1]["ideologies"]
    
    shifts = []
    for r in range(len(initial_ideologies)):
        shift = [
            final_ideologies[r][i] - initial_ideologies[r][i] 
            for i in range(3)
        ]
        shifts.append(shift)
        
    survival_duration = episodes * 100 
    
    print("=== WorldSim End-of-Simulation Analysis ===")
    print(f"Total Episodes Run: {episodes}")
    print(f"Total Survival Duration (Steps): {survival_duration}")
    print(f"Total Collapse Events: {total_collapses}")
    print("Average Ideology Shifts (Final - Initial):")
    for r, shift in enumerate(shifts):
        print(f"  Region {r}: Coop: {shift[0]:+.2f} | Agg: {shift[1]:+.2f} | Sust: {shift[2]:+.2f}")
        
    print("\nTrade Network Trust (Final):")
    trust_net = data[-1].get("trust_network", {})
    for u in trust_net:
        out = []
        for v in trust_net[u]:
            out.append(f"{v}: {trust_net[u][v].get('trust', 0):.2f}")
        print(f"  Region {u} -> {', '.join(out)}")
        
if __name__ == "__main__":
    analyze()
