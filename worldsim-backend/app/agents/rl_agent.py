"""
Reinforcement Learning-based Regional Agents
"""
import numpy as np
from typing import Tuple, List, Dict, Any
from collections import deque
import random

class DQNAgent:
    """Simple Q-Learning Agent for regional agent decision making"""
    
    def __init__(self, state_size: int = 12, action_size: int = 9, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-table (simplified): use random weights for approximation
        self.q_values = np.random.randn(action_size) * 0.01
        self.memory = deque(maxlen=2000)
        self.loss_history = []
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Use state-action approximation
        q_estimates = self.q_values + np.sum(state) * 0.1
        return np.argmax(q_estimates)
    
    def replay(self, batch_size: int = 32):
        """Experience replay training"""
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.memory, batch_size)
        
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                next_q = np.max(self.q_values + np.sum(next_state) * 0.1)
                target = reward + self.gamma * next_q
            

            old_q = self.q_values[action]
            self.q_values[action] += self.learning_rate * (target - old_q)
            loss = abs(target - old_q)
            total_loss += loss
        
        avg_loss = total_loss / batch_size
        self.loss_history.append(avg_loss)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return avg_loss


class RegionalAgent:
    """AI Agent governing a region using RL"""
    
    # Action space: (resource allocation, trade strategy)
    ACTIONS = {
        0: {'name': 'Invest in Food', 'water': -50, 'food': 100, 'energy': -20},
        1: {'name': 'Invest in Energy', 'water': -30, 'food': -20, 'energy': 150},
        2: {'name': 'Invest in Water', 'water': 200, 'food': -50, 'energy': -50},
        3: {'name': 'Increase Production', 'pop_growth': 0.05, 'energy': -100},
        4: {'name': 'Trade (Aggressive)', 'trade_intensity': 0.8, 'energy': -30},
        5: {'name': 'Trade (Moderate)', 'trade_intensity': 0.5, 'energy': -15},
        6: {'name': 'Conserve Resources', 'conservation': 0.3, 'growth': -0.02},
        7: {'name': 'Invest in Development', 'dev_increase': 0.05, 'energy': -80},
        8: {'name': 'Do Nothing', 'passive': True}
    }
    
    def __init__(self, region_id: str, dqn_agent: DQNAgent):
        self.region_id = region_id
        self.dqn = dqn_agent
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.episode_steps = 0
        
    def get_state_vector(self, region_state: Dict[str, Any]) -> np.ndarray:
        """Convert region state to normalized state vector for RL"""
        resources = region_state['resources']
        env_state = np.array([
            resources['water'] / 2000.0,      # 0: normalized water
            resources['food'] / 2000.0,       # 1: normalized food
            resources['energy'] / 2000.0,     # 2: normalized energy
            resources['land'] / 1000.0,       # 3: normalized land
            region_state['population'] / 1000.0,  # 4: normalized population
            region_state['development_level'],    # 5: dev level (0-1)
            region_state['stability'],            # 6: stability (0-1)
            region_state['temperature'] / 50.0,   # 7: normalized temp
            region_state['rainfall'] / 200.0,     # 8: normalized rainfall
            region_state['disaster_risk'],        # 9: disaster risk
            len(region_state['trade_partners']) / 10.0,  # 10: normalized trade partners
            region_state['growth_rate'] * 100  # 11: growth rate
        ], dtype=np.float32)
        
        return np.clip(env_state, 0, 1)
    
    def decide_action(self, region_state: Dict[str, Any], training: bool = True) -> Tuple[int, str]:
        """Use RL to decide action"""
        state_vec = self.get_state_vector(region_state)
        action_idx = self.dqn.act(state_vec, training=training)
        action_name = self.ACTIONS[action_idx]['name']
        return action_idx, action_name
    
    def calculate_reward(self, prev_resources: Dict, curr_resources: Dict, 
                       stability: float, population: int) -> float:
        """Calculate reward based on state change"""
        # Resource balance reward
        resource_health = (
            (curr_resources['water'] / 2000.0) +
            (curr_resources['food'] / 2000.0) +
            (curr_resources['energy'] / 2000.0)
        ) / 3.0
        
        # Stability reward
        stability_reward = stability * 2
        
        # Population growth reward
        pop_reward = min(population / 500.0, 1.0) * 0.5
        
        # Prevent starvation penalty
        starvation_penalty = 0
        if curr_resources['food'] < 100:
            starvation_penalty = -5
        if curr_resources['energy'] < 100:
            starvation_penalty -= 3
        
        total_reward = (resource_health * 3 + stability_reward + pop_reward + starvation_penalty)
        return float(np.clip(total_reward, -10, 10))
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool):
        """Learn from experience"""
        self.dqn.remember(state, action, reward, next_state, done)
        self.reward_history.append(reward)
        self.action_history.append(action)
        
        loss = self.dqn.replay(batch_size=32)
        return loss
