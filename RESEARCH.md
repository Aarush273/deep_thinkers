# WorldSim - Research & Analysis Guide

## Research Applications

WorldSim enables investigation of complex resource dynamics and emergent agent behavior. This guide outlines how to design and execute research studies.

## Research Question Framework

### Question 1: Emergent Strategies
**"What strategies naturally emerge from RL agents in resource-constrained environments?"**

#### Methodology
1. Run simulation for 200-500 cycles
2. Collect strategy data: `/simulation/analysis`
3. Analyze action frequency per region
4. Compare top strategies across regions
5. Identify patterns and clusters

#### Analysis Metrics
```python
strategy_execution_count = sum(action_history)
strategy_success = avg(reward_history) per strategy
strategy_adoption_time = first_cycle where strategy emerges
strategy_persistence = frequency across total actions
```

#### Expected Findings
- Some regions become specialists (e.g., food producers)
- Generalist vs. specialist trade-offs emerge
- Learning plateaus indicated by consistent strategies
- Occasionally maladaptive strategies persist due to local optima

#### Code to Extract Data
```bash
# Get all region analyses
curl 'http://localhost:8000/simulation/analysis' | jq '.regions'

# Parse strategy distributions
# For each region, top_strategies shows frequency
```

---

### Question 2: Trade Network Dynamics
**"How do trade networks self-organize and affect regional stability?"**

#### Methodology
1. Track trade network evolution over time
2. Compare regions WITH vs. WITHOUT trade
3. Measure dependency ratio (trade ÷ total resources)
4. Analyze collapse rates by connectivity

#### Experimental Design
**Condition A (Control)**: Normal simulation with trade
**Condition B (Treatment)**: Disable trade execution

To disable trade:
```python
# In app/simulation/engine.py, step() method:
# Comment out trade execution section:
# for region_a_id, trade_partners in [...]:
#     for region_b_id, strength in trade_partners.items():
#         if np.random.random() > 0.5:
#             self.trade_network.execute_trade(...)
```

#### Analysis Metrics
```
Trade dependence = (resources gained from trade) / (total resources)
Network density = num_edges / max_possible_edges
Regional isolation impact = survival_time(isolated) - survival_time(connected)
Cascading collapse rate = regions_collapsed / time_to_collapse
```

#### Expected Findings
- High trade density → increased stability (specialization benefits)
- Moderate trade → optimal (diversity + redundancy)
- Low trade → fragile (isolation risk)
- Disruption of key traders → cascade failures

#### Data Collection
```bash
# Monitor trade network each cycle
curl 'http://localhost:8000/trade-network' | jq '.edges | length'

# Track in spreadsheet:
# Cycle, NumTrades, NumRegions, AvgTradeStrength, PopulationTrend
```

---

### Question 3: Disaster Resilience
**"Which strategies make regions most resilient to environmental shocks?"**

#### Methodology
1. Run baseline simulation (event probability = 0)
2. Run treatment simulation (event probability = 0.5)
3. Compare development trajectories
4. Analyze adaptation to recurring events

#### Experimental Design
**Baseline**: `np.random.random() > 0.15` → change to `> 0.99` (almost no events)
**Treatment**: `np.random.random() > 0.05` (5× more events)

#### Analysis Metrics
```
Event survival rate = regions_surviving / total_events
Recovery time = cycles_to_regain_prev_resources
Adaptive capacity = improvement in stability post-event
Strategy variance = std_dev(actions) - higher = more adaptive
```

#### Code Modification
```python
# In app/simulation/engine.py:
def _generate_climatic_event(self):
    # For high-event condition:
    if np.random.random() > 0.05:  # was 0.15
        return None
```

#### Expected Findings
- Conservative strategies (Conserve Resources) rise in high-event conditions
- Diversified portfolios outperform specialists
- Some regions fail to adapt and collapse
- Learning agents eventually optimize for specific event types

---

### Question 4: Development vs. Growth Trade-offs
**"Is there an optimal development/growth balance?"**

#### Methodology
1. Initialize regions with varied development_level
2. Track population trajectories
3. Analyze long-term sustainability
4. Measure development impact on resource production

#### Experimental Conditions
- **Condition A**: All regions start with dev_level = 0.3
- **Condition B**: All regions start with dev_level = 0.7
- **Condition C**: Mixed (0.3-0.7)

Code modification:
```python
# In app/simulation/engine.py, _initialize_world():
region = RegionState(
    # ...
    development_level=0.7,  # Change initial value
)
```

#### Analysis Metrics
```
Carrying capacity = max(population) reached
Collapse rate = regions_failed / total_regions
Resource efficiency = population_sustenance / resource_consumption
Development ROI = (final_dev - initial_dev) / energy_invested
```

#### Expected Findings
- Development costs energy but enables food production
- Very low dev → starvation
- Very high dev → over-investment waste
- Optimal range emerges around 0.5-0.7

---

### Question 5: Population Dynamics & Carrying Capacity
**"What determines regional carrying capacity?"**

#### Methodology
1. Fixed resources at various levels
2. Measure max sustainable population
3. Track cycles to collapse at different capacities
4. Model relationship: $Population = f(Resources, Development)$

#### Analysis Metrics
```
Carrying capacity = max(population) before decline
Stability index = time_at_max_pop / total_cycles
Overshoot magnitude = max(pop) - sustainable_pop
Recovery pattern = population trend post-crash
```

#### Expected Findings
- Non-linear relationship with development
- Trade enables higher carrying capacity
- Regional carrying capacity converges through learning
- Overshoots and oscillations before equilibrium

---

## Experimental Procedure

### Step 1: Define Hypothesis
```
H0 (Null): Strategy A and B have equal impact on stability
H1 (Alt):  Strategy A results in higher stability than B
α = 0.05 (significance level)
```

### Step 2: Design Experiment
```
Samples:  10 replicates per condition (different seeds)
Duration: 500 cycles per replicate
Metrics:  [stability, population, collapse_rate, dev_level]
```

### Step 3: Data Collection Script

```python
# save_file: experiment.py
import subprocess
import json
import time

results = []

for condition in ['baseline', 'high_events', 'no_trade']:
    for seed in range(1, 11):  # 10 replicates
        # Reset with custom seed
        # (Would need to modify engine for this)
        
        # Run simulation
        for cycle in range(500):
            resp = requests.post('http://localhost:8000/simulation/step')
            
            if cycle % 50 == 0:  # Collect every 50 cycles
                state = requests.get('http://localhost:8000/simulation/state').json()
                stats = requests.get('http://localhost:8000/simulation/statistics').json()
                
                results.append({
                    'condition': condition,
                    'seed': seed,
                    'cycle': cycle,
                    'population': stats['total_population'],
                    'stability': stats['avg_stability'],
                    'dev': stats['avg_development'],
                    'trades': stats['trade_connections']
                })
        
        # Reset for next replicate
        requests.post('http://localhost:8000/simulation/reset')
        time.sleep(1)

# Save results
with open('experiment_results.json', 'w') as f:
    json.dump(results, f)
```

### Step 4: Statistical Analysis

```python
# analysis.py
import json
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load data
with open('experiment_results.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Analyze by condition
for condition in df['condition'].unique():
    subset = df[df['condition'] == condition]['stability']
    print(f"{condition}: mean={subset.mean():.3f}, std={subset.std():.3f}")

# T-test between conditions
baseline = df[df['condition'] == 'baseline']['stability']
high_events = df[df['condition'] == 'high_events']['stability']

t_stat, p_value = stats.ttest_ind(baseline, high_events)
print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")

# Visualization
df.groupby('condition')['stability'].mean().plot(kind='bar')
plt.ylabel('Average Stability')
plt.xlabel('Condition')
plt.savefig('results.png')
```

---

## Analysis Tools

### Export Cycle Data

```bash
# Get last 200 cycles
curl 'http://localhost:8000/simulation/history?limit=200' > cycles.json

# Pretty print
cat cycles.json | jq .
```

### Parse in Python

```python
import json
import pandas as pd

with open('cycles.json') as f:
    data = json.load(f)

cycles = data['cycles']
df = pd.json_normalize([
    {
        'cycle': cycle['cycle'],
        **{f"{rid}_pop": region['population'] 
           for rid, region in cycle['regions'].items()}
    }
    for cycle in cycles
])

print(df.head())
df.to_csv('population_trends.csv')
```

### Visualization Queries

```python
# Plot population over time
import matplotlib.pyplot as plt

populations = [cycle['regions']['region_0']['population'] 
               for cycle in cycles]
plt.plot(populations)
plt.ylabel('Population')
plt.xlabel('Cycle')
plt.title('Region 0 Population Trajectory')
plt.show()
```

---

## Research Findings Template

### Study: [Research Question]

**Hypothesis**: [State hypothesis]

**Method**:
- Conditions: [Experimental conditions]
- Duration: [# cycles]
- Replicates: [# seeds]
- Metrics: [What was measured]

**Results**:
```
[Summary table of findings]
```

**Statistical Analysis**:
```
[T-tests, confidence intervals, p-values]
```

**Interpretation**: [What patterns emerged?]

**Limitations**: [Constraints of the simulation]

**Implications**: [Real-world applicability]

---

## Simulation Constraints

⚠️ **Important limitations to acknowledge**:

1. **Simplified World**: Only 4 resources, 6-12 regions
2. **Deterministic Dynamics**: Fixed equations for resource generation
3. **Limited Agent Complexity**: Simple DQN, not human-like reasoning
4. **No Communications**: Agents can't negotiate, only trade
5. **Markov Assumption**: Current state independent of history
6. **Homogeneous Agents**: All regions have identical RL architecture

---

## Publishing Results

### Suggested Conference/Journal Topics
- Emergent Behavior in Resource Networks
- Reinforcement Learning for Policy Design
- Complex Adaptive Systems in Climate Scenarios
- Agent-Based Modeling of Resource Conflict

### Recommended Figures
1. Population trajectories (line plot)
2. Trade network evolution (snapshot at cycles 0, 100, 300)
3. Strategy distributions (heatmap of actions over time)
4. Stability distribution (box plot by condition)
5. Event impact analysis (before/after recovery curves)

---

## Next Research Extensions

1. **Multi-Objective RL**: Agents optimize multiple goals
2. **Communication Protocol**: Agents can send messages
3. **Policy Interventions**: Government actions that affect agents
4. **Learning Heterogeneity**: Different agent learning rates
5. **Geographic Constraints**: Trade costs based on distance
6. **Sector Specialization**: Agriculture, Mining, Manufacturing
7. **Human-in-the-Loop**: Player controls a region

---

**Start your research!** Choose a question, run your experiment, and discover what WorldSim reveals about adaptive agents under pressure.
