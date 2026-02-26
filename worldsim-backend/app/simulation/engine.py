"""
World Simulation Engine - Main orchestrator
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from datetime import datetime
from app.core.resources import ResourcePool, RegionState
from app.agents.rl_agent import RegionalAgent, DQNAgent

class ClimaticEvent:
    """Represents environmental events affecting world"""
    
    EVENT_TYPES = {
        'drought': {'water': -500, 'food': -300, 'description': 'Severe Drought'},
        'flood': {'water': 200, 'food': -200, 'land': -100, 'description': 'Flooding'},
        'heatwave': {'energy': -200, 'food': -150, 'description': 'Heatwave'},
        'coldsnap': {'energy': -400, 'food': -100, 'description': 'Cold Snap'},
        'harvest': {'food': 300, 'description': 'Bountiful Harvest'},
        'plague': {'population_loss': 0.1, 'description': 'Plague Outbreak'}
    }
    
    def __init__(self, event_type: str, affected_regions: List[str], severity: float = 1.0):
        self.event_type = event_type
        self.affected_regions = affected_regions
        self.severity = severity
        self.timestamp = datetime.now()
    
    def apply(self, region_state: RegionState) -> Dict[str, float]:
        """Apply event to region"""
        if region_state.region_id not in self.affected_regions:
            return {}
        
        event_data = self.EVENT_TYPES.get(self.event_type, {})
        effects = {}
        
        for resource, change in event_data.items():
            if resource == 'population_loss':
                region_state.population = int(region_state.population * (1 - change * self.severity))
            elif hasattr(region_state.resources, resource):
                current = getattr(region_state.resources, resource)
                setattr(region_state.resources, resource, 
                       max(0, current + change * self.severity))
                effects[resource] = change * self.severity
        
        return effects


class TradeNetwork:
    """Manages trade relationships between regions"""
    
    def __init__(self):
        self.trade_graph = nx.DiGraph()
        self.trade_history: List[Dict] = []
    
    def add_region(self, region_id: str):
        """Add region to trade network"""
        self.trade_graph.add_node(region_id)
    
    def establish_trade(self, region_a: str, region_b: str, strength: float = 0.5):
        """Establish trade relationship"""
        self.trade_graph.add_edge(region_a, region_b, weight=strength)
        self.trade_graph.add_edge(region_b, region_a, weight=strength)
    
    def get_trading_partners(self, region_id: str) -> Dict[str, float]:
        """Get trading partners and strength"""
        partners = {}
        if region_id in self.trade_graph:
            for neighbor in self.trade_graph.neighbors(region_id):
                edge_data = self.trade_graph.get_edge_data(region_id, neighbor)
                partners[neighbor] = edge_data.get('weight', 0.5)
        return partners
    
    def execute_trade(self, region_a_state: RegionState, region_b_state: RegionState,
                     trade_strength: float = 0.5) -> Tuple[Dict, Dict]:
        """Execute trade between regions"""
        # Simplified trade: exchange surpluses
        trade_amount = trade_strength * 50
        
        # Region A gives food, gets energy
        region_a_state.resources.food -= trade_amount
        region_b_state.resources.energy -= trade_amount * 0.8  # Loss in transit
        
        # Region B gives energy, gets food
        region_b_state.resources.energy += trade_amount * 0.9  # Profit
        region_a_state.resources.food += trade_amount * 0.8
        
        trade_event = {
            'region_a': region_a_state.region_id,
            'region_b': region_b_state.region_id,
            'strength': trade_strength,
            'timestamp': datetime.now()
        }
        self.trade_history.append(trade_event)
        
        return {}, {}


class WorldSimulation:
    """Main simulation engine"""
    
    def __init__(self, num_regions: int = 6, seed: int = 42):
        np.random.seed(seed)
        self.num_regions = num_regions
        self.current_cycle = 0
        self.regions: Dict[str, RegionState] = {}
        self.agents: Dict[str, RegionalAgent] = {}
        self.trade_network = TradeNetwork()
        self.event_history: List[ClimaticEvent] = []
        self.action_history: List[Dict] = []
        self.cycle_history: List[Dict] = []
        
        self._initialize_world()
    
    def _initialize_world(self):
        """Initialize world with regions and agents"""
        region_names = [
            'Northern Plains', 'Forest Kingdom', 'Coastal Republic',
            'Desert Alliance', 'Mountain Federation', 'Valley States'
        ]
        
        for i in range(self.num_regions):
            region_id = f"region_{i}"
            region_name = region_names[i] if i < len(region_names) else f"Region {i}"
            
            # Create region with random environmental factors
            region = RegionState(
                region_id=region_id,
                name=region_name,
                resources=ResourcePool(
                    water=np.random.uniform(800, 1200),
                    food=np.random.uniform(800, 1200),
                    energy=np.random.uniform(800, 1200),
                    land=1000
                ),
                population=np.random.randint(80, 120),
                development_level=np.random.uniform(0.3, 0.7),
                temperature=np.random.uniform(15, 25),
                rainfall=np.random.uniform(80, 150)
            )
            
            self.regions[region_id] = region
            
            # Create RL agent for region
            dqn = DQNAgent(state_size=12, action_size=9)
            agent = RegionalAgent(region_id, dqn)
            self.agents[region_id] = agent
            
            # Add to trade network
            self.trade_network.add_region(region_id)
        
        # Establish initial trade relationships
        self._establish_initial_trades()
    
    def _establish_initial_trades(self):
        """Create initial trade network"""
        # Random initial trades with neighbors
        regions_list = list(self.regions.keys())
        for i, region_a in enumerate(regions_list):
            for j in range(i + 1, min(i + 3, len(regions_list))):
                region_b = regions_list[j]
                if np.random.random() > 0.3:  # 70% chance of trade
                    self.trade_network.establish_trade(region_a, region_b, 
                                                       strength=np.random.uniform(0.3, 0.8))
    
    def _generate_climatic_event(self) -> Optional[ClimaticEvent]:
        """Randomly generate climatic events"""
        if np.random.random() > 0.15:  # 15% chance per cycle
            return None
        
        event_types = list(ClimaticEvent.EVENT_TYPES.keys())
        event_type = np.random.choice(event_types)
        
        num_affected = np.random.randint(1, max(2, self.num_regions // 2))
        affected_regions = list(np.random.choice(
            list(self.regions.keys()), size=num_affected, replace=False
        ))
        
        severity = np.random.uniform(0.5, 1.5)
        return ClimaticEvent(event_type, affected_regions, severity)
    
    def _update_basic_resources(self, region: RegionState):
        """Update region resources based on natural processes"""
        # Natural precipitation/water generation
        water_generation = region.rainfall * 5
        region.resources.replenish(water=water_generation)
        
        # Food production based on development and land
        food_production = region.development_level * region.population * 2
        region.resources.replenish(food=food_production)
        
        # Energy consumption
        energy_consumption = region.population * 5
        region.resources.deplete(energy=energy_consumption)
        
        # Population changes
        if region.resources.food > region.population * 8:
            region.population = int(region.population * (1 + region.growth_rate))
        elif region.resources.food < region.population * 3:
            region.population = int(region.population * 0.98)
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation cycle"""
        cycle_data = {
            'cycle': self.current_cycle,
            'timestamp': datetime.now().isoformat(),
            'regions': {},
            'events': [],
            'actions': []
        }
        
        # Update basic resources for all regions
        for region_id, region in self.regions.items():
            self._update_basic_resources(region)
        
        # Apply climatic events
        event = self._generate_climatic_event()
        if event:
            self.event_history.append(event)
            for region_id, region in self.regions.items():
                event.apply(region)
            cycle_data['events'].append({
                'type': event.event_type,
                'affected_regions': event.affected_regions,
                'severity': event.severity
            })
        
        # Agent decision making and resource allocation
        for region_id, region in self.regions.items():
            agent = self.agents[region_id]
            state_before = region.resources.get_as_dict().copy()
            
            # Get agent action
            action_idx, action_name = agent.decide_action(region.get_state_dict())
            action_effects = RegionalAgent.ACTIONS[action_idx]
            
            # Apply action effects
            if 'water' in action_effects:
                region.resources.deplete(water=-action_effects['water'])
            if 'food' in action_effects:
                region.resources.deplete(food=-action_effects['food'])
            if 'energy' in action_effects:
                region.resources.deplete(energy=-action_effects['energy'])
            if 'growth' in action_effects:
                region.growth_rate += action_effects['growth']
            if 'pop_growth' in action_effects:
                region.population = int(region.population * (1 + action_effects['pop_growth']))
            if 'dev_increase' in action_effects:
                region.development_level = min(1.0, region.development_level + action_effects['dev_increase'])
            
            # Calculate reward
            state_after = region.resources.get_as_dict()
            reward = agent.calculate_reward(state_before, state_after, region.stability, region.population)
            
            # Store experience and learn
            state_vec = agent.get_state_vector(region.get_state_dict())
            next_state_vec = agent.get_state_vector(region.get_state_dict())
            done = region.is_critical()
            agent.learn(state_vec, action_idx, reward, next_state_vec, done)
            
            cycle_data['regions'][region_id] = region.get_state_dict()
            cycle_data['actions'].append({
                'region_id': region_id,
                'action': action_name,
                'reward': reward
            })
        
        # Execute trades
        for region_a_id, trade_partners in [(rid, self.trade_network.get_trading_partners(rid)) 
                                           for rid in self.regions.keys()]:
            for region_b_id, strength in trade_partners.items():
                if region_a_id < region_b_id:  # Avoid duplicate trades
                    if np.random.random() > 0.5:  # 50% chance to trade each cycle
                        region_a = self.regions[region_a_id]
                        region_b = self.regions[region_b_id]
                        self.trade_network.execute_trade(region_a, region_b, strength)
        
        self.cycle_history.append(cycle_data)
        self.current_cycle += 1
        
        return cycle_data
    
    def get_world_state(self) -> Dict[str, Any]:
        """Get current complete world state"""
        return {
            'cycle': self.current_cycle,
            'regions': {rid: region.get_state_dict() for rid, region in self.regions.items()},
            'trade_network': {
                'edges': [
                    {'source': u, 'target': v, 'weight': data.get('weight', 0.5)}
                    for u, v, data in self.trade_network.trade_graph.edges(data=True)
                ]
            },
            'events': [
                {
                    'type': event.event_type,
                    'affected_regions': event.affected_regions,
                    'severity': event.severity,
                    'timestamp': event.timestamp.isoformat()
                }
                for event in self.event_history[-10:]  # Last 10 events
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate simulation statistics"""
        total_population = sum(region.population for region in self.regions.values())
        avg_stability = np.mean([region.stability for region in self.regions.values()])
        avg_dev = np.mean([region.development_level for region in self.regions.values()])
        
        # Count collapsed regions
        collapsed = sum(1 for region in self.regions.values() if region.is_critical())
        
        # Average agent learning progress
        avg_learning = np.mean([len(agent.reward_history) for agent in self.agents.values()])
        
        return {
            'cycle': self.current_cycle,
            'total_population': int(total_population),
            'avg_stability': float(avg_stability),
            'avg_development': float(avg_dev),
            'collapsed_regions': collapsed,
            'active_regions': self.num_regions - collapsed,
            'trade_connections': self.trade_network.trade_graph.number_of_edges(),
            'avg_learning_steps': float(avg_learning)
        }
