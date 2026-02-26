# WorldSim Technical Documentation

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      React Frontend (Port 3000)                   │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │ WorldMap     │ Region       │ Statistics   │ Trade        │  │
│  │ Component    │ Dashboard    │ Panel        │ Network      │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ EventLog Component                                         │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                          ↕ (HTTP/REST)
┌─────────────────────────────────────────────────────────────────┐
│               FastAPI Backend (Port 8000)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ API Routes Layer (/simulation, /regions, /events)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ WorldSimulation Engine                                    │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Regions (6+) - DQN Agents - Trade Network - Events │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Core Models: ResourcePool, RegionState, RL Agents      │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Core Models (`app/core/resources.py`)

#### ResourcePool
```python
class ResourcePool:
    water: float      # Current water level
    food: float       # Current food level
    energy: float     # Current energy level
    land: float       # Land (semi-static)
    
    # Constraints
    max_water: float = 2000.0
    max_food: float = 2000.0
    max_energy: float = 2000.0
    max_land: float = 1000.0
```

**Key Methods**:
- `deplete(water, food, energy, land)` - Consume resources with zero-floor
- `replenish(water, food, energy, land)` - Add resources with cap enforcement
- `get_total_value()` - Calculate weighted total (land×10 weighted higher)
- `is_critical()` - Check if any resource < 100

#### RegionState
```python
class RegionState:
    region_id: str         # Unique identifier
    name: str              # Display name
    resources: ResourcePool
    population: int        # Current population
    development_level: float  # 0-1 scale
    growth_rate: float     # Per-cycle growth
    stability: float       # 0-1, affects agent decisions
    trade_partners: Dict[str, float]  # neighboring_id -> strength
    
    # Environment
    temperature: float     # Celsius
    rainfall: float        # mm/month
    disaster_risk: float   # Probability per cycle
```

### 2. RL Agents (`app/agents/rl_agent.py`)

#### DQNAgent (PyTorch)
```python
class DQNAgent(nn.Module):
    # Architecture: 12 input → 128 → 128 → 9 output
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Returns Q-values for each action
    
    def act(self, state: np.ndarray, training: bool) -> int:
        # Epsilon-greedy action selection
        # Exploration-exploitation trade-off
    
    def remember(self, state, action, reward, next_state, done):
        # Store in replay buffer (max 2000)
    
    def replay(self, batch_size: int):
        # Experience replay training
        # Sample batch and compute DQN loss
```

**Training Loop**:
1. Select action with ε-greedy policy
2. Execute action in simulation
3. Get reward and next state
4. Store (s, a, r, s', done) in replay buffer
5. Sample batch and compute: loss = ||Q(s,a) - (r + γ·max_a'Q(s',a'))||²
6. Backward pass and optimizer step
7. Decay epsilon

#### RegionalAgent
```python
class RegionalAgent:
    ACTIONS = {
        0: {'name': 'Invest in Food', ...},
        1: {'name': 'Invest in Energy', ...},
        # ... 7 more actions
        8: {'name': 'Do Nothing', ...}
    }
    
    def get_state_vector(region_state: Dict) -> np.ndarray:
        # Normalize 12 features to [0, 1]
    
    def decide_action(region_state, training) -> Tuple[int, str]:
        # Call DQN.act() and return action index + name
    
    def calculate_reward(prev_resources, curr_resources, stability, pop) -> float:
        # Multi-faceted reward:
        # - Resource health (0-3)
        # - Stability bonus (0-2)
        # - Population growth (0-0.5)
        # - Starvation penalties (-5 to -3)
```

### 3. Simulation Engine (`app/simulation/engine.py`)

#### ClimaticEvent
```python
class ClimaticEvent:
    event_type: str          # drought, flood, heatwave, etc.
    affected_regions: List[str]
    severity: float          # 0.5-1.5 multiplier
    
    def apply(self, region: RegionState):
        # Modifies region resources based on event
```

#### TradeNetwork (NetworkX Graph)
```python
class TradeNetwork:
    trade_graph: nx.DiGraph  # Directed graph
    
    def establish_trade(region_a, region_b, strength):
        # Bidirectional edge with weight
    
    def execute_trade(region_a_state, region_b_state, strength):
        # Exchange: A gives food → gets energy
        # B gives energy → gets food
        # 20% loss in transit
```

#### WorldSimulation (Main Orchestrator)
```python
class WorldSimulation:
    num_regions: int                    # Usually 6
    current_cycle: int                  # Simulation counter
    regions: Dict[str, RegionState]     # All regions
    agents: Dict[str, RegionalAgent]    # RL agents
    trade_network: TradeNetwork
    event_history: List[ClimaticEvent]
    cycle_history: List[Dict]           # Full cycle data
    
    def step() -> Dict:
        # Execute one full simulation cycle
        # Returns: {cycle, timestamp, regions, events, actions}
    
    def _update_basic_resources(region):
        # Water: +rainfall×5
        # Food: +development×population×2
        # Energy: -population×5
        # Population: growth or decline based on food
    
    def get_world_state() -> Dict:
        # Current complete state
    
    def get_statistics() -> Dict:
        # Derived metrics for UI
```

**Simulation Cycle Flow**:
```
1. Update basic resources for all regions
2. Generate climatic event (15% probability)
3. For each region:
   a. Get state vector (12-dim normalized)
   b. Agent decides action via DQN
   c. Apply action effects to resources
   d. Calculate reward
   e. Store experience, train DQN
4. Execute trades (50% per edge)
5. Record cycle data
6. Return cycle_data
```

### 4. FastAPI Routes (`app/api/routes.py`)

**Core Endpoints**:

```python
POST /simulation/step
# Request: empty
# Response: {
#   'cycle': int,
#   'regions': Dict[region_id -> state],
#   'events': List[{type, affected_regions, severity}],
#   'actions': List[{region_id, action, reward}]
# }

GET /simulation/state
# Current world state snapshot

GET /simulation/statistics
# Global metrics: population, stability, trades, etc.

GET /simulation/history?limit=50
# Recent cycle history array

POST /simulation/reset
# Reset to initial state

GET /regions
# All regions state

GET /regions/{region_id}
# Specific region + agent stats

GET /trade-network
# Trade graph nodes and edges

GET /events?limit=50
# Event history

GET /simulation/analysis
# Strategic analysis per region + sustainability metrics
```

## State Vector Normalization

Agent state input (12 dimensions):
```
[0] water / 2000.0           → [0, 1]
[1] food / 2000.0            → [0, 1]
[2] energy / 2000.0          → [0, 1]
[3] land / 1000.0            → [0, 1]
[4] population / 1000.0      → [0, 0.1+]
[5] development_level        → [0, 1]
[6] stability                → [0, 1]
[7] temperature / 50.0       → [0, 1]
[8] rainfall / 200.0         → [0, 1]
[9] disaster_risk            → [0, 1]
[10] trade_partners / 10.0   → [0, 0.6]
[11] growth_rate * 100       → [0, 5+]
```

All clipped to [0, 1] range for consistency.

## Reward Structure

```python
resource_health = avg(water/2000, food/2000, energy/2000)
stability_bonus = stability × 2
population_bonus = min(population/500, 1.0) × 0.5

penalties = 0
if food < 100:
    penalties -= 5
if energy < 100:
    penalties -= 3

total_reward = (resource_health × 3) + stability_bonus + population_bonus + penalties
# Range: [-10, 10]
```

## Action Space Details

| ID | Name | Water | Food | Energy | Effect |
|----|------|-------|------|--------|--------|
| 0 | Invest Food | -50 | +100 | -20 | Boost food production |
| 1 | Invest Energy | -30 | -20 | +150 | Boost energy gen |
| 2 | Invest Water | +200 | -50 | -50 | Boost water reserves |
| 3 | Increase Prod | - | - | -100 | +5% pop growth |
| 4 | Trade Aggressive | - | - | -30 | 80% trade intensity |
| 5 | Trade Moderate | - | - | -15 | 50% trade intensity |
| 6 | Conserve | - | - | - | -2% growth |
| 7 | Dev Invest | - | - | -80 | +5% dev level |
| 8 | Do Nothing | - | - | - | Passive |

## Frontend Data Flow

```
React App (App.js)
│
├─ useEffect() → axios.get(/simulation/state)
├─ setInterval(step, cycleSpeed)
│  └─ axios.post(/simulation/step)
│
├─ WorldMap
│  └─ Renders: region cards with pie charts
│     onClick → setSelectedRegion()
│
├─ RegionDashboard
│  ├─ axios.get(/regions/{region_id})
│  └─ Displays: resource bars, agent stats
│
├─ StatisticsPanel
│  └─ Displays: cycle #, population, stability barchart
│
├─ TradeNetwork
│  └─ Displays: trade edges, strength bars
│
└─ EventLog
   └─ Displays: recent events with emoji/severity
```

## Performance Considerations

### Computational Complexity
- **Simulation Step**: O(num_regions × (batch_size + trade_edges))
  - Per region: state vector calc (O(1)), DQN forward (O(256)), backward (O(256))
  - Trades: O(num_regions²) but sparse in practice
  - Overall for 6 regions: ~10-50ms per cycle

- **Memory**: ~500MB
  - DQN weights: ~1.5MB × 6 agents = 9MB
  - Replay buffers: ~2000 × (state + reward) = ~2MB × 6 = 12MB
  - Cycle history (100 cycles): ~50MB

### Optimization Tips
1. Batch DQN training every N steps (vs. every step)
2. Prune old cycle history
3. Use float32 instead of float64
4. WebSocket for real-time vs. polling

## Testing the System

### Manual Test Cycle
```bash
# Terminal 1: Start backend
cd worldsim-backend
python main.py

# Terminal 2: Start frontend
cd worldsim-frontend
npm start

# In browser at http://localhost:3000:
# 1. Click ▶ Play
# 2. Observe regions updating
# 3. Click region card → see details
# 4. Check trade network forming
# 5. Wait for event (15% per cycle)
```

### API Test
```bash
# Get initial state
curl http://localhost:8000/simulation/state | jq

# Step once
curl -X POST http://localhost:8000/simulation/step | jq

# Get analysis
curl http://localhost:8000/simulation/analysis | jq '.regions'
```

## Debugging

### Enable verbose logging
In `app/api/routes.py`, add:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check agent learning
```python
# In API or test script
agent = simulation.agents['region_0']
print(f"Epsilon: {agent.dqn.epsilon}")
print(f"Actions taken: {len(agent.action_history)}")
print(f"Avg reward: {sum(agent.reward_history) / len(agent.reward_history)}")
```

### Inspect state vector
```python
region = simulation.regions['region_0']
agent = simulation.agents['region_0']
state_vec = agent.get_state_vector(region.get_state_dict())
print(state_vec)  # Should be 12 values in [0,1]
```

## Extension Points

1. **Add new action**: Extend `RegionalAgent.ACTIONS` dict
2. **New event type**: Add to `ClimaticEvent.EVENT_TYPES`
3. **Different RL algorithm**: Replace `DQNAgent` with PPO/A3C
4. **Custom reward**: Modify `RegionalAgent.calculate_reward()`
5. **Multi-agent communication**: Extend TradeNetwork with message protocol
6. **Persistent save**: Add SQLAlchemy models + DB

---

**Last Updated**: February 26, 2026
