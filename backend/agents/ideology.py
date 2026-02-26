import numpy as np
import random

class IdeologyTracker:
    def __init__(self, num_regions=4):
        self.num_regions = num_regions
        
        # Ideology vectors for each region: [cooperation, aggression, sustainability]
        self.ideologies = np.random.uniform(0.1, 0.9, size=(num_regions, 3))
        
        # Track history per region to update ideology
        self.action_history = [[] for _ in range(num_regions)]
        self.reward_history = [[] for _ in range(num_regions)]
        
    def get_ideology(self, region_id):
        return self.ideologies[region_id]
        
    def apply_bias(self, region_id, ppo_action):
        """
        With some probability proportional to the bias, override the PPO action 
        with an ideology-driven action.
        Actions:
        0 = Conserve, 1 = Balanced, 2 = Aggressive, 3 = Trade, 4 = Tech
        """
        coop, agg, sust = self.ideologies[region_id]
        
        r = random.random()
        
        # Thresholds: ensure total probability isn't overwhelming, max sum ~0.6
        if r < coop * 0.2:
            return 3  # Trade
        elif r < (coop * 0.2 + agg * 0.2):
            return 2  # Aggressive
        elif r < (coop * 0.2 + agg * 0.2 + sust * 0.2):
            return 0  # Conserve
            
        # Otherwise, stick to PPO action
        return ppo_action
        
    def add_experience(self, region_id, action, reward):
        self.action_history[region_id].append(action)
        self.reward_history[region_id].append(reward)
        
        if len(self.action_history[region_id]) >= 10:
            self.update_ideology(region_id)
            
    def update_ideology(self, region_id):
        """
        If trade actions led to positive rewards: increase cooperation_bias
        If aggressive actions caused instability: decrease aggression_bias
        If conserve actions improved long-term stability: increase sustainability_bias
        """
        actions = self.action_history[region_id]
        rewards = self.reward_history[region_id]
        
        coop, agg, sust = self.ideologies[region_id]
        
        for a, r in zip(actions, rewards):
            if a == 3: # Trade
                if r > 0: coop += 0.05
                else: coop -= 0.02
            elif a == 2: # Aggressive
                if r < 0: agg -= 0.05
                else: agg += 0.02
            elif a == 0: # Conserve
                if r > 0: sust += 0.05
                else: sust -= 0.02
                
        # Clamp values
        self.ideologies[region_id] = np.clip([coop, agg, sust], 0.0, 1.0)
        
        self.action_history[region_id].clear()
        self.reward_history[region_id].clear()
