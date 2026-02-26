import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
from backend.agents.ideology import IdeologyTracker

CLIMATE_NORMAL = 0
CLIMATE_DROUGHT = 1
CLIMATE_FLOOD = 2

class WorldEnv(gym.Env):
    """
    WorldSim: A multi-agent RL simulation focusing on resource scarcity,
    adaptive ideologies, trade networks, and climate shocks.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, num_regions=4):
        super().__init__()
        self.num_regions = num_regions
        
        # PPO outputs actions for all 4 regions simultaneously
        self.action_space = spaces.MultiDiscrete([5] * num_regions)
        
        # State per region: water, food, energy, land, population, avg_neighbor, climate, avg_trust
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_regions, 8), dtype=np.float32)
        
        self.ideology_tracker = IdeologyTracker(num_regions)
        self.trade_graph = nx.complete_graph(num_regions)
        for u, v in self.trade_graph.edges():
            self.trade_graph[u][v]['trust'] = 0.5
            
        self.step_count = 0
        self.max_steps = 100
        self.global_climate = CLIMATE_NORMAL
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.global_climate = CLIMATE_NORMAL
        
        # Resources state logic 
        # Array shape (4, 7): 0:water, 1:food, 2:energy, 3:land, 4:pop, 5:stability, 6:tech
        self.resources = np.zeros((self.num_regions, 7), dtype=np.float32)
        for r in range(self.num_regions):
            self.resources[r, 0] = 100.0   # water
            self.resources[r, 1] = 100.0   # food
            self.resources[r, 2] = 100.0   # energy
            self.resources[r, 3] = 1000.0  # land (used as resource capacity limit)
            self.resources[r, 4] = 10.0    # population
            self.resources[r, 5] = 1.0     # stability
            self.resources[r, 6] = 1.0     # tech level (multiplier)
            
        self.ideology_tracker = IdeologyTracker(self.num_regions)
        self.trade_graph = nx.complete_graph(self.num_regions)
        for u, v in self.trade_graph.edges():
            self.trade_graph[u][v]['trust'] = 0.5
            
        # Base regeneration rates
        self.base_regens = np.array([ [15.0, 15.0, 10.0, 0, 0, 0, 0] for _ in range(self.num_regions) ], dtype=np.float32)
        
        # For info tracking
        self.global_collapses = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.zeros((self.num_regions, 8), dtype=np.float32)
        for r in range(self.num_regions):
            water = self.resources[r, 0]
            food = self.resources[r, 1]
            energy = self.resources[r, 2]
            land = self.resources[r, 3]
            pop = self.resources[r, 4]
            
            # Avg neighbor resources
            nb_res = []
            avg_trust = 0
            for n in self.trade_graph.neighbors(r):
                nb_res.append(self.resources[n, 0] + self.resources[n, 1] + self.resources[n, 2])
                avg_trust += self.trade_graph[r][n]['trust']
                
            avg_n_res = np.mean(nb_res) if nb_res else 0
            avg_trust = avg_trust / (self.num_regions - 1) if self.num_regions > 1 else 0
            
            obs[r] = [water, food, energy, land, pop, avg_n_res, float(self.global_climate), avg_trust]
            
        return obs
        
    def step(self, actions):
        self.step_count += 1
        
        # 1. Update global climate (Markov-style)
        if self.global_climate == CLIMATE_NORMAL:
            self.global_climate = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
        elif self.global_climate == CLIMATE_DROUGHT:
            self.global_climate = np.random.choice([0, 1], p=[0.3, 0.7])
        elif self.global_climate == CLIMATE_FLOOD:
            self.global_climate = np.random.choice([0, 2], p=[0.4, 0.6])
            
        climate_modifier = 1.0
        if self.global_climate == CLIMATE_DROUGHT:
            climate_modifier = 0.5
        elif self.global_climate == CLIMATE_FLOOD:
            climate_modifier = 1.2
            
        rewards = np.zeros(self.num_regions)
        actual_actions = np.zeros(self.num_regions, dtype=int)
        
        # Apply actions and calculate individual impact
        for r in range(self.num_regions):
            # Apply ideology bias overlay
            actual_action = self.ideology_tracker.apply_bias(r, actions[r])
            actual_actions[r] = actual_action
            
            consumption_multiplier = 1.0
            if actual_action == 0:  # Conserve
                consumption_multiplier = 0.6
            elif actual_action == 1:  # Balanced
                consumption_multiplier = 1.0
            elif actual_action == 2:  # Aggressive
                consumption_multiplier = 1.6
            elif actual_action == 4:  # Tech
                self.resources[r, 6] += 0.05
                
            pop = self.resources[r, 4]
            w_usage = pop * 0.5 * consumption_multiplier
            f_usage = pop * 0.5 * consumption_multiplier
            e_usage = pop * 0.3 * consumption_multiplier
            
            tech_mod = self.resources[r, 6]
            
            # Regeneration logic bounds to land capacity
            capacity = self.resources[r, 3]
            
            self.resources[r, 0] = min(capacity, self.resources[r, 0] + self.base_regens[r, 0] * climate_modifier * tech_mod)
            
            f_mod = climate_modifier if self.global_climate != CLIMATE_FLOOD else 0.7
            self.resources[r, 1] = min(capacity, self.resources[r, 1] + self.base_regens[r, 1] * f_mod * tech_mod)
            
            self.resources[r, 2] = min(capacity, self.resources[r, 2] + self.base_regens[r, 2] * tech_mod)
            
            # Consumption logic
            self.resources[r, 0] -= w_usage
            self.resources[r, 1] -= f_usage
            self.resources[r, 2] -= e_usage
            
            # Overuse penalty: Reduce regeneration if > 80% usage capacity
            total_usage = w_usage + f_usage + e_usage
            if total_usage > 0.8 * (self.base_regens[r, 0] + self.base_regens[r, 1] + self.base_regens[r, 2]):
                self.base_regens[r, 0] *= 0.95
                self.base_regens[r, 1] *= 0.95
                
            # Collapse penalty
            collapse = False
            for res_idx in [0, 1, 2]:
                if self.resources[r, res_idx] <= 0:
                    self.resources[r, res_idx] = 0
                    self.resources[r, 5] -= 0.3  # Severity stability penalty
                    collapse = True
                    
            if collapse:
                self.resources[r, 4] *= 0.85 # pop decrease
                self.global_collapses += 1
            else:
                self.resources[r, 4] += pop * 0.05 * consumption_multiplier  # growth
                
            # Keep stability in boundaries
            self.resources[r, 5] = max(0.0, min(1.0, self.resources[r, 5] + 0.01))
            
            # Reward
            survival_bonus = 2.0 if not collapse else -5.0
            pop_growth_reward = 0.5 * (self.resources[r, 4] - pop)
            sustainability_score = 1.0 if actual_action == 0 else 0.0
            stability_penalty = -2.0 if self.resources[r, 5] < 0.4 else 0.0
            
            # Substantial penalty on hitting 0
            zero_penalty = -5.0 if collapse else 0.0
            
            rewards[r] = survival_bonus + pop_growth_reward + (0.3 * sustainability_score) + stability_penalty + zero_penalty
            
            self.ideology_tracker.add_experience(r, actual_action, rewards[r])
            
        # 3. Trade logic mapping actions: 3 = Propose trade
        traders = np.where(actual_actions == 3)[0]
        if len(traders) >= 2:
            # Pair them linearly
            for i in range(len(traders)):
                u = traders[i]
                v = traders[(i + 1) % len(traders)]
                
                trust = self.trade_graph[u][v]['trust']
                if trust > 0.3:
                    # Trade logic: exchange surplus for deficit
                    # Assuming r0 needs food, r1 needs water
                    for surplus_item, deficit_item in [(0, 1), (1, 0)]:
                        if self.resources[u, surplus_item] > 50 and self.resources[v, deficit_item] > 50:
                            # Execute swap
                            self.resources[u, surplus_item] -= 10
                            self.resources[v, surplus_item] += 10
                            
                            self.resources[v, deficit_item] -= 10
                            self.resources[u, deficit_item] += 10
                            
                            self.trade_graph[u][v]['trust'] = min(1.0, trust + 0.1)
                            rewards[u] += 1.0
                            rewards[v] += 1.0
                            break
                    else:
                        # Trade fails due to lack of complementary surplus
                        self.trade_graph[u][v]['trust'] = max(0.0, trust - 0.05)
                else:
                    self.trade_graph[u][v]['trust'] = max(0.0, trust - 0.05)
                    
        done = self.step_count >= self.max_steps
        
        info = {
            "region_rewards": rewards,
            "region_actions": actual_actions,
            "climate": self.global_climate,
            "ideologies": [self.ideology_tracker.get_ideology(r) for r in range(self.num_regions)],
            "resources": self.resources.copy(),
            "global_collapses": self.global_collapses,
            "trust_network": nx.to_dict_of_dicts(self.trade_graph)
        }
        
        return self._get_obs(), float(np.sum(rewards)), done, False, info
