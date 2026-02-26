# WorldSim - Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd worldsim-backend

# Create virtual environment (one-time)
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (CMD):
venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API server
python main.py
```

✅ **Backend running**: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### Step 2: Frontend Setup (New Terminal)

```bash
# Navigate to frontend directory
cd worldsim-frontend

# Install dependencies (one-time)
npm install

# Start React dev server
npm start
```

✅ **Frontend running**: Opens automatically at `http://localhost:3000`

## 🎮 Using WorldSim

### Main Dashboard
- **Play/Pause**: Control simulation flow
- **Step**: Execute one cycle at a time
- **Reset**: Restart simulation fresh
- **Speed slider**: Control cycle speed

### Regions
- Click any region card for detailed analysis
- Watch colors change:
  - 🟢 Green: Healthy (stability > 60%)
  - 🟡 Yellow: Fragile (stability < 60%)
  - 🔴 Red: Critical (low resources)

### Monitoring
1. **Statistics Panel**: Overall world health
2. **Trade Network**: Active trade routes
3. **Region Details**: Click region → see resources, agent behavior
4. **Event Log**: Environmental events that occurred

## 📊 What to Observe

### First Run (0-50 cycles)
- Regions establishing trade
- Population growth/decline
- Resource balance forming

### Mid-Run (50-200 cycles)
- Emergent agent strategies
- Specialist regions (e.g., one focuses on food)
- Trade network stabilizing

### Long-Run (200+ cycles)
- Stable vs. collapsing regions
- Repeated event impacts
- Learned resilience patterns

## 🔧 Customize Simulation

### Change number of regions
**File**: `worldsim-backend/app/api/routes.py`

```python
# Find: WorldSimulation(num_regions=6)
# Change to:
WorldSimulation(num_regions=8)  # More regions
```

### Change cycle speed
**In UI**: Use the speed slider (100-2000ms)

### Modify resource generation
**File**: `worldsim-backend/app/simulation/engine.py`

```python
def _update_basic_resources(self, region):
    # Change water_generation
    water_generation = region.rainfall * 5  # Increase multiplier
    
    # Change food_production
    food_production = region.development_level * region.population * 2  # Adjust
```

### Make events more/less common
**File**: `worldsim-backend/app/simulation/engine.py`

```python
def _generate_climatic_event(self):
    # Current: 15% chance per cycle
    if np.random.random() > 0.15:  # Change 0.15 to desired probability
        return None
```

## 🐛 Troubleshooting

### "Failed to fetch world state"
- Check backend is running: `http://localhost:8000/docs`
- Check CORS: Backend has `allow_origins=["*"]`

### Port already in use
**Port 8000 (backend)**:
```bash
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -i :8000
kill -9 <PID>
```

**Port 3000 (frontend)**:
- React will auto-suggest alternative port

### Slow simulation
- Reduce batch size in RL training (app/agents/rl_agent.py, line ~replay)
- Or reduce frequency of training (replay less often)

### Agents learning too slowly
- Increase learning rate: `learning_rate=0.001` (from 0.0001)
- Decrease epsilon decay: `epsilon_decay=0.99` (from 0.995)

## 📈 Running Analysis

### Get Strategic Analysis
```bash
curl http://localhost:8000/simulation/analysis | jq
```

Shows for each region:
- Top strategies adopted
- Learning progress
- Most recent actions

### Export Cycle Data
```bash
# Get last 100 cycles
curl 'http://localhost:8000/simulation/history?limit=100' > cycle_history.json
```

Use for statistical analysis, plotting trends, etc.

## 💡 Research Ideas

1. **Trade Dependency**: Remove trade, compare stability
2. **Event Sensitivity**: Increase event frequency, observe adaptation
3. **Development Levels**: Start all regions at different dev levels
4. **Population Control**: Test if limiting growth helps stability
5. **Resource Ratios**: Change max resource limits per region
6. **Agent Heterogeneity**: Give agents different learning rates
7. **Intervention**: Add "government aid" mechanism, measure effectiveness

## 📝 Expected Behavior

### Healthy Simulation
```
Cycle 0-30:   Regions stabilizing, trade forming
Cycle 30-100: Growth phase, population increasing, stability 0.6-0.8
Cycle 100+:   Equilibrium or slow evolution, occasional region stress
```

### Signs of Issues
- All regions collapsing (rapid red) → resource initialization too low
- No events ever happening → probability too low
- Agents not learning → learning rate too low, epsilon decaying too fast
- Population infinite growth → growth rate too high

## 🎯 Next Steps

1. Run simulation for 100+ cycles using Play button
2. Click different regions to see agent decision history
3. Review Analysis endpoint for strategy insights
4. Modify resource generation or event rates
5. Compare multiple runs with different seeds
6. Export data and analyze trends
7. Implement custom rewards or actions

## 📚 Documentation

- **README.md**: Complete project overview
- **TECHNICAL.md**: Deep technical details
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Code**: Well-commented source files

## ✨ Tips

- **Seed for reproducibility**: Modify `WorldSimulation(seed=42)` in routes.py
- **Time travel**: Can't rewind, but can reset and run again
- **Export data**: Use cycle history for Excel/Python analysis
- **Pause for inspection**: Click Pause, click region, examine state
- **Run multiple instances**: Start fresh simulations to compare

---

**Ready to explore?** Open `http://localhost:3000` and click ▶ Play!
