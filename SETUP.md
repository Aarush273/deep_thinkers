# WorldSim - Setup Instructions for Team

## System Requirements

- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **Git** (for version control)
- **RAM**: 2GB minimum
- **Disk**: 500MB free

## Step-by-Step Setup

### 1. Clone/Download Project

```bash
# If using git
git clone <repository>
cd worldsim

# Or navigate to the extracted folder
```

### 2. Backend Setup

```bash
cd worldsim-backend

# Create Python virtual environment
python -m venv venv

# Activate it
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows Command Prompt:
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, numpy, networkx; print('✓ All dependencies installed')"
```

### 3. Frontend Setup

```bash
# In a NEW terminal/PowerShell window
cd worldsim-frontend

# Install Node dependencies
npm install

# Verify installation
npm -v && node -v
```

### 4. Start Servers

**Terminal 1 - Backend:**
```bash
cd worldsim-backend
# Activate venv (if not already)
source venv/bin/activate  # or appropriate command
python main.py
```
Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Terminal 2 - Frontend:**
```bash
cd worldsim-frontend
npm start
```
Expected output:
```
Compiled successfully!

You can now view worldsim-frontend in the browser.
  
  Local:            http://localhost:3000
```

## Verification

### Check Backend
```bash
# In a 3rd terminal:
curl http://localhost:8000/simulation/state | jq .cycle
# Should return: 0 (initial cycle)
```

### Check Frontend
- Browser should show WorldSim dashboard
- Six region cards visible
- Control buttons functional

## File Structure Reference

```
worldsim/
├── README.md                      # Overview
├── QUICKSTART.md                  # 5-min setup
├── TECHNICAL.md                   # Architecture details
├── SETUP.md                       # This file
│
├── worldsim-backend/
│   ├── main.py                    # FastAPI entry point
│   ├── requirements.txt            # Python dependencies
│   └── app/
│       ├── core/
│       │   ├── resources.py       # Resource & region models
│       │   └── __init__.py
│       ├── agents/
│       │   ├── rl_agent.py        # DQN & agent logic
│       │   └── __init__.py
│       ├── simulation/
│       │   ├── engine.py          # Main simulation orchestrator
│       │   └── __init__.py
│       ├── api/
│       │   ├── routes.py          # FastAPI endpoints
│       │   └── __init__.py
│       └── __init__.py
│
└── worldsim-frontend/
    ├── package.json               # NPM configuration
    ├── public/
    │   └── index.html             # HTML entry point
    └── src/
        ├── App.js                 # Main React component
        ├── App.css                # Global styles
        ├── index.js               # React entry
        ├── index.css
        └── components/
            ├── WorldMap.js & .css         # Region visualization
            ├── RegionDashboard.js & .css  # Region details
            ├── StatisticsPanel.js & .css  # Global stats
            ├── TradeNetwork.js & .css     # Trade viz
            └── EventLog.js & .css         # Event display
```

## Development Workflow

### Working on Backend
```bash
cd worldsim-backend
source venv/bin/activate
# Edit files in app/
# Changes auto-reload in FastAPI
```

### Working on Frontend
```bash
cd worldsim-frontend
# Edit files in src/
# Changes auto-reload in React dev server
```

### Running Tests
```bash
# Backend unit tests (to be added)
cd worldsim-backend
python -m pytest tests/

# Frontend tests (to be added)
cd worldsim-frontend
npm test
```

## Troubleshooting

### Issue: "Python not found"
**Solution**: Install Python 3.8+ from python.org, ensure it's in PATH

### Issue: "pip install fails"
**Solution**:
```bash
# Upgrade pip
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: "npm install fails"
**Solution**:
```bash
# Clear npm cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Issue: Port 8000 already in use
**Solution**:
```bash
# Change port in main.py:
# Change: uvicorn.run(app, host="0.0.0.0", port=8000)
# To:     uvicorn.run(app, host="0.0.0.0", port=8001)

# Or kill existing process:
# macOS/Linux: lsof -i :8000 | kill -9 $(awk 'NR!=1 {print $2}')
# Windows: netstat -ano | findstr :8000, then taskkill /PID <pid> /F
```

### Issue: Frontend can't connect to backend
**Solution**:
Check API_BASE in worldsim-frontend/src/App.js:
```javascript
const API_BASE = 'http://localhost:8000';
```
Should match your backend address

### Issue: DQN training very slow
**Solution**: Reduce training frequency
```python
# In app/agents/rl_agent.py, learn() method:
# loss = self.dqn.replay(batch_size=32)  # Default
# Change to:
loss = self.dqn.replay(batch_size=16)   # Smaller batches
# Or call replay() less frequently
```

## First Run Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000
- [ ] Can see 6 region cards
- [ ] Play/Pause buttons respond
- [ ] Can click region for details
- [ ] Statistics updating each cycle
- [ ] Events appearing in log (~15% per cycle)
- [ ] Trade network showing connections
- [ ] No console errors (F12 → Console)

## Performance Tips

1. **Reduce simulation speed** if CPU usage high
2. **Limit cycle history** to last 50 cycles (see TECHNICAL.md)
3. **Use production React build** for deployment:
   ```bash
   cd worldsim-frontend
   npm run build
   ```

## Deployment

### For Presentation/Demo
```bash
# Terminal 1:
cd worldsim-backend && source venv/bin/activate && python main.py

# Terminal 2:
cd worldsim-frontend && npm start
```

### For Production
- Build React: `npm run build`
- Serve with: `python -m http.server` in build/
- Use production Uvicorn configuration
- Add environment variables for API endpoint

## Environment Variables

Create `.env` file in `worldsim-backend/` if needed:
```
API_PORT=8000
API_HOST=0.0.0.0
LOG_LEVEL=INFO
```

Load in `main.py`:
```python
from dotenv import load_dotenv
load_dotenv()
port = int(os.getenv('API_PORT', 8000))
```

## IDE Setup (VS Code)

### Extensions to Install
- Python (Microsoft)
- Flask Snippet (Evan Chamran)  
- ES7+ React/Redux/React-Native snippets
- REST Client

### Debug Backend
Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["main:app", "--reload"],
      "jinja": true,
      "cwd": "${workspaceFolder}/worldsim-backend"
    }
  ]
}
```

## Common Commands

```bash
# Backend
python main.py                    # Start server
pip install <package>            # Add dependency
pip freeze > requirements.txt    # Update requirements

# Frontend  
npm start                        # Dev server
npm run build                    # Production build
npm install <package>           # Add dependency
npm test                        # Run tests

# API Testing
curl http://localhost:8000/simulation/state   # Get state
curl -X POST http://localhost:8000/simulation/step  # Step
```

## Documentation Navigation

| Document | Purpose |
|----------|---------|
| README.md | Project overview, features, applications |
| QUICKSTART.md | 5-minute setup & basic usage |
| **SETUP.md** | **This file - full team setup** |
| TECHNICAL.md | Architecture, API details, debugging |

## Getting Help

1. Check TECHNICAL.md for architecture details
2. Review code comments (docstrings)
3. Check API docs: `http://localhost:8000/docs`
4. Test with curl: `curl http://localhost:8000/simulation/state | jq`

## Next Steps After Setup

1. **Play through full simulation**: Run 100+ cycles
2. **Analyze strategies**: Check `/simulation/analysis`
3. **Modify parameters**: Adjust resource generation, event rates
4. **Add features**: Implement new actions or events
5. **Export data**: Save cycle history for analysis
6. **Research**: Study emergent strategies and patterns

---

**Setup complete!** 🎉

Open `http://localhost:3000` and click ▶ Play to begin.

Questions? Refer to TECHNICAL.md or check code comments.
