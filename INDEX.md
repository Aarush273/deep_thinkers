# WorldSim Project Index

## 📚 Documentation Map

Welcome to **WorldSim** - an Adaptive Resource Scarcity & Agent Strategy Simulator for the SIT Hackathon.

### Quick Navigation

| Document | Purpose | Read If... |
|----------|---------|-----------|
| **QUICKSTART.md** | 5-minute setup guide | You want to run it NOW |
| **SETUP.md** | Full team installation guide | You're setting up for the first time |
| **README.md** | Project overview & features | You want to understand what WorldSim does |
| **TECHNICAL.md** | Architecture & deep dive | You need to modify code or debug |
| **RESEARCH.md** | Research methodology & experiments | You're doing analysis/studies |
| **This file** | Project index | You're getting oriented |

---

## 🚀 Getting Started (Choose Your Path)

### Path 1: Run It Yourself (5 minutes)
1. Read: **QUICKSTART.md**
2. Execute: Backend setup → Frontend setup
3. Open: `http://localhost:3000`
4. Click: Play button

### Path 2: Understand It First
1. Read: **README.md** (architecture overview)
2. Skim: **TECHNICAL.md** (system design)
3. Then follow: **QUICKSTART.md**

### Path 3: Research/Analysis
1. Read: **README.md** (overview)
2. Study: **TECHNICAL.md** (how it works)
3. Follow: **RESEARCH.md** (experimental design)
4. Use: API endpoints for data collection

### Path 4: Team Development
1. First person: Follow **SETUP.md**
2. All team members: Follow **QUICKSTART.md**
3. Developers: Refer to **TECHNICAL.md**
4. Researchers: Refer to **RESEARCH.md**

---

## 📁 Project Structure

```
worldsim/
│
├─ 📄 README.md              → Main project documentation
├─ 📄 QUICKSTART.md          → 5-min setup guide
├─ 📄 SETUP.md               → Full setup instructions
├─ 📄 TECHNICAL.md           → Architecture & internals
├─ 📄 RESEARCH.md            → Research methodology
├─ 📄 INDEX.md               → This file
├─ 📄 .gitignore             → Git configuration
│
├─ 📁 worldsim-backend/      → Python FastAPI backend
│   ├─ main.py               → Application entry point
│   ├─ requirements.txt      → Python dependencies
│   └─ app/
│       ├─ core/
│       │   └─ resources.py  → Resource & region models
│       ├─ agents/
│       │   └─ rl_agent.py   → DQN agents & learning
│       ├─ simulation/
│       │   └─ engine.py     → World simulation orchestrator
│       └─ api/
│           └─ routes.py     → REST API endpoints
│
└─ 📁 worldsim-frontend/     → React.js frontend
    ├─ package.json          → NPM configuration
    ├─ public/
    │   └─ index.html        → HTML entry point
    └─ src/
        ├─ App.js            → Main React component
        ├─ App.css           → Global styles
        ├─ index.js
        ├─ index.css
        └─ components/
            ├─ WorldMap.js & .css         → Region cards
            ├─ RegionDashboard.js & .css  → Region details
            ├─ StatisticsPanel.js & .css  → Global stats
            ├─ TradeNetwork.js & .css     → Trade visualization
            └─ EventLog.js & .css         → Event log
```

---

## 🎯 Feature Overview

### What WorldSim Does
- **Simulates** 6+ regions with finite resources (water, food, energy, land)
- **Deploys** AI agents using Deep Q-Learning to make strategic decisions
- **Evolves** trade networks dynamically based on region needs
- **Applies** environmental events (droughts, floods, plagues, etc.)
- **Tracks** emerging strategies, sustainability patterns, and collapse dynamics
- **Visualizes** real-time world state, statistics, and agent behavior

### Key Features
✅ Reinforcement learning agents that adapt over time
✅ Dynamic trade networks with emergent partnerships
✅ Environmental events with realistic impacts
✅ Population dynamics based on resource availability
✅ Interactive web-based dashboard with live updates
✅ Detailed metrics on agent learning and strategies
✅ REST API for data extraction and analysis
✅ Reproducible simulations with seed control

---

## 🔬 Research Applications

WorldSim enables research in:

1. **Emergent Strategy Discovery**
   - What strategies do agents develop?
   - Are there consistent patterns?
   - How do strategies evolve over time?

2. **Trade Network Dynamics**
   - How do trade relationships self-organize?
   - What triggers network disruption?
   - How resilient are interdependent systems?

3. **Resource Conflict & Cooperation**
   - When do agents cooperate vs. compete?
   - How do environmental crises affect behavior?
   - Can agents learn to prevent collapse?

4. **Resilience & Sustainability**
   - Which strategies are most resilient?
   - What causes cascading failures?
   - How do agents adapt to recurring shocks?

5. **Policy Analysis**
   - What interventions stabilize systems?
   - How do population controls affect outcomes?
   - What resource distribution is optimal?

See **RESEARCH.md** for detailed experimental methodology.

---

## 💻 Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **ML**: PyTorch (DQN implementation)
- **Data**: NumPy (numerical computing)
- **Networks**: NetworkX (graph processing)
- **Server**: Uvicorn (ASGI)

### Frontend
- **Framework**: React 18
- **Charts**: Recharts (data visualization)
- **HTTP**: Axios (API calls)
- **Styling**: CSS3

### Architecture Pattern
- **REST API**: Stateless endpoints
- **Real-time**: HTTP polling (WebSocket ready)
- **State Management**: React hooks + local state
- **Data Flow**: Backend → API → Frontend components

---

## 🚀 Quick Commands

### Backend
```bash
cd worldsim-backend
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate.bat         # Windows
pip install -r requirements.txt
python main.py                    # Runs on port 8000
```

### Frontend
```bash
cd worldsim-frontend
npm install
npm start                         # Opens port 3000
```

### API Testing
```bash
curl http://localhost:8000/docs                    # API documentation
curl http://localhost:8000/simulation/state        # Current state
curl -X POST http://localhost:8000/simulation/step # Execute cycle
curl http://localhost:8000/simulation/analysis     # Analyze strategies
```

---

## 📊 Key Concepts

### Simulation State
- **Cycle**: Each simulation step represents one time period
- **Region**: A geographical/political entity with resources and population
- **Agent**: AI controller using RL to manage a region
- **Trade**: Resource exchange between regions (20% loss in transit)

### Resource Types
- **Water**: Generated from rainfall, consumed gradually
- **Food**: Produced by development level, consumed by population
- **Energy**: Required for activities, consumed based on actions
- **Land**: Static resource, supports agriculture/development

### Event System
- **Probability**: ~15% chance per cycle
- **Types**: Drought, Flood, Heatwave, Cold Snap, Harvest, Plague
- **Impact**: Varies by severity (0.5-1.5 multiplier)
- **Scope**: Affects 1-3 random regions

### Agent Decision Space
**9 Actions**: 
- Invest in Food/Energy/Water
- Increase Production
- Trade (aggressive/moderate)
- Conserve Resources
- Invest in Development
- Do Nothing

**Reward**: -10 to +10 based on:
- Resource health (food, water, energy levels)
- Regional stability
- Population growth
- Penalties for starvation

---

## 📈 Expected Behavior

### Healthy Simulation Progression
```
Cycles 0-30:    Initialization, trade network forming
Cycles 30-100:  Growth phase, population increasing
Cycles 100-300: Equilibrium seeking, strategy refinement
Cycles 300+:    Stable patterns or gradual decline/specialization
```

### Visual Indicators
- 🟢 **Green cards**: Stable regions (stability > 60%, resources adequate)
- 🟡 **Yellow cards**: Stressed regions (stability 30-60%, resources tight)
- 🔴 **Red cards**: Critical regions (stability < 30%, resources critical)

### Strategy Markers
- **Early**: Chaotic actions (high epsilon, exploration)
- **Mid**: Specialization emerges (regions focus on specific actions)
- **Late**: Convergence or collapse (strategies lock in or fail)

---

## 🔍 Analysis Workflow

### 1. Collect Data
```bash
# Export simulation history
curl 'http://localhost:8000/simulation/history?limit=500' > data.json

# Get strategic analysis
curl 'http://localhost:8000/simulation/analysis' > analysis.json
```

### 2. Process Data
```python
import json, pandas as pd

with open('data.json') as f:
    cycles = json.load(f)['cycles']

# Create dataframe
df = pd.json_normalize([
    {'cycle': c['cycle'], 
     'region_0_pop': c['regions']['region_0']['population']}
    for c in cycles])

# Analyze
print(df.describe())
```

### 3. Visualize & Interpret
```python
import matplotlib.pyplot as plt

df['region_0_pop'].plot()
plt.ylabel('Population')
plt.xlabel('Cycle')
plt.savefig('pop_trend.png')
```

---

## 🐛 Troubleshooting Checklist

| Issue | Solution | Doc Reference |
|-------|----------|---|
| "Command not found: python" | Install Python 3.8+ | SETUP.md |
| Port 8000 in use | Kill process or change port | SETUP.md |
| Virtual env won't activate | Check shell type (PowerShell vs CMD) | SETUP.md |
| npm install fails | Run `npm cache clean --force` | SETUP.md |
| Frontend can't reach API | Check API_BASE in App.js | SETUP.md |
| Agents learning too slowly | Adjust learning rate in rl_agent.py | TECHNICAL.md |
| Simulation runs very slowly | Reduce batch size, disable logging | TECHNICAL.md |

---

## 📖 Documentation Glossary

### Key Terms
- **DQN**: Deep Q-Network - neural network learns state→action values
- **Epsilon-Greedy**: Exploration vs exploitation strategy
- **Replay Buffer**: Memory of past experiences for training
- **Carrying Capacity**: Maximum population resources can sustain
- **Trade Intensity**: Percentage of resources exchanged in trade
- **Stability**: 0-1 measure of regional health and governance
- **Development Level**: 0-1 measure of technological advancement

### API Terminology
- **State**: Current snapshot of world (resources, populations, etc.)
- **Action**: Decision made by agent (e.g., "Invest in Food")
- **Reward**: Numerical feedback (-10 to +10) for agent learning
- **Episode**: Full simulation run (e.g., 500 cycles)
- **Cycle**: Single timestep in simulation

---

## 🎓 For Students/Learners

### Understanding the Code

**Start here**:
1. `worldsim-backend/app/core/resources.py` - Data models
2. `worldsim-backend/app/simulation/engine.py` - Main logic
3. `worldsim-backend/app/agents/rl_agent.py` - AI logic
4. `worldsim-frontend/src/components/` - UI components

**Key concepts to learn**:
- How state vectors represent world
- How DQN learns Q-values
- How rewards shape behavior
- How trade network evolves

### Common Modifications

Adding a new action:
```python
# In rl_agent.py, RegionalAgent.ACTIONS dict:
5: {'name': 'My Action', 'water': -50, ...}
```

Changing event probability:
```python
# In engine.py, _generate_climatic_event():
if np.random.random() > 0.20:  # Change from 0.15 to 0.20
```

---

## 📞 Support & Questions

### Getting Help
1. **Setup issues**: See SETUP.md
2. **How to use**: See QUICKSTART.md  
3. **How it works**: See TECHNICAL.md or code comments
4. **Research questions**: See RESEARCH.md
5. **API details**: Visit `http://localhost:8000/docs`

### Debugging
- Check console logs (browser F12)
- Check backend logs (terminal running main.py)
- Use curl to test API endpoints
- Print statements in Python code

---

## 🎉 Next Steps

### Immediate (Right Now)
1. Read QUICKSTART.md
2. Run the setup commands
3. Click Play and observe

### Short Term (First Hour)
4. Observe 100+ cycles
5. Click different regions
6. Check Analysis endpoint
7. Explore API docs

### Medium Term (First Day)
8. Modify a parameter (event rate, resource amount)
9. Run comparison simulations
10. Export and analyze data
11. Try a research question from RESEARCH.md

### Longer Term
12. Implement new features or actions
13. Conduct full research experiment
14. Document findings
15. Present results

---

## 📜 Project Metadata

- **Project**: WorldSim v1.0
- **Event**: SIT Hackathon
- **Date Created**: February 26, 2026
- **Technologies**: Python, React, PyTorch, FastAPI, NetworkX
- **Team**: [Your Team Name]
- **Status**: Ready for exploration and research

---

## 🏆 What Makes WorldSim Special

✨ **Fully Autonomous Agents**: No hard-coded rules, pure RL
✨ **Dynamic Environments**: Trade networks and events evolve
✨ **Real-time Visualization**: See strategies emerge live
✨ **Research-Ready**: Extract data for analysis
✨ **Hackathon-Complete**: Functional system, not a demo
✨ **Highly Extensible**: Easy to add features and experiments

---

## 📚 Recommended Reading Order

For **Complete Understanding**:
1. README.md (overview)
2. QUICKSTART.md (setup)
3. Run simulation and observe
4. TECHNICAL.md (architecture)
5. Explore code with comments

For **Research/Analysis**:
1. README.md (context)
2. QUICKSTART.md (setup)
3. RESEARCH.md (methodology)
4. Run experiments
5. TECHNICAL.md (debugging)

For **Development**:
1. SETUP.md (installation)
2. QUICKSTART.md (basic usage)
3. TECHNICAL.md (architecture)
4. Examine code structure
5. Start implementing features

---

## 🚀 Ready to Begin?

```
$ cd worldsim-backend && python main.py
$ # In new terminal:
$ cd worldsim-frontend && npm start
$ # Open http://localhost:3000 and click ▶ Play
```

**Welcome to WorldSim!** 🌍

---

*Last Updated: February 26, 2026*
*For questions, refer to the appropriate documentation file above.*
