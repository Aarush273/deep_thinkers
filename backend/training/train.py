import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.env.world_env import WorldEnv

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.history = []
        
    def _on_step(self):
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            
            def convert_to_native(obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                elif isinstance(obj, dict): return {str(k): convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list): return [convert_to_native(v) for v in obj]
                else: return obj
                
            history_entry = {
                "episode": len(self.history) + 1,
                "collapses": convert_to_native(info.get("global_collapses", 0)),
                "climate": convert_to_native(info.get("climate", 0)),
                "ideologies": convert_to_native(info.get("ideologies", [])),
                "resources": convert_to_native(info.get("resources", [])),
                "trust_network": convert_to_native(info.get("trust_network", {}))
            }
            self.history.append(history_entry)
            
            os.makedirs("output", exist_ok=True)
            with open("output/metrics.json", "w") as f:
                json.dump(self.history, f)
                
        return True

def main():
    print("Initializing WorldEnv...")
    env = WorldEnv(num_regions=4)
    
    print("Instantiating PPO Agent...")
    # small batch size and short train time for local execution viability
    model = PPO("MlpPolicy", env, verbose=1, n_steps=200, batch_size=50)
    
    callback = MetricsCallback()
    
    print("Training for 2000 steps...")
    model.learn(total_timesteps=2000, callback=callback)
    
    print("Training Complete. Metrics saved to output/metrics.json")
    print("\nRunning Analyzer...")
    os.system("python backend/analysis/analyzer.py")
    
if __name__ == "__main__":
    main()
