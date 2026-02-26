import React, { useState, useEffect } from 'react';
import axios from 'axios';
import WorldMap from './components/WorldMap';
import RegionDashboard from './components/RegionDashboard';
import TradeNetwork from './components/TradeNetwork';
import EventLog from './components/EventLog';
import StatisticsPanel from './components/StatisticsPanel';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [worldState, setWorldState] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [cycleSpeed, setCycleSpeed] = useState(500);
  const [cycleHistory, setCycleHistory] = useState([]);

  // Fetch world state
  const fetchWorldState = async () => {
    try {
      const response = await axios.get(`${API_BASE}/simulation/state`);
      setWorldState(response.data);
    } catch (error) {
      console.error('Failed to fetch world state:', error);
    }
  };

  // Fetch statistics
  const fetchStatistics = async () => {
    try {
      const response = await axios.get(`${API_BASE}/simulation/statistics`);
      setStatistics(response.data);
    } catch (error) {
      console.error('Failed to fetch statistics:', error);
    }
  };

  // Step simulation
  const stepSimulation = async () => {
    try {
      const response = await axios.post(`${API_BASE}/simulation/step`);
      setCycleHistory([response.data, ...cycleHistory.slice(0, 99)]);
      await fetchWorldState();
      await fetchStatistics();
    } catch (error) {
      console.error('Failed to step simulation:', error);
    }
  };

  // Auto-step simulation
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(stepSimulation, cycleSpeed);
    return () => clearInterval(interval);
  }, [isRunning, cycleSpeed]);

  // Initial fetch
  useEffect(() => {
    fetchWorldState();
    fetchStatistics();
  }, []);

  return (
    <div className="App">
      <header className="app-header">
        <h1>🌍 WorldSim - Adaptive Resource Scarcity & Agent Strategy Simulator</h1>
        <div className="controls">
          <button 
            onClick={() => setIsRunning(!isRunning)}
            className={`btn ${isRunning ? 'btn-stop' : 'btn-play'}`}
          >
            {isRunning ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={stepSimulation} className="btn">
            ⏭ Step
          </button>
          <button 
            onClick={async () => {
              await axios.post(`${API_BASE}/simulation/reset`);
              fetchWorldState();
              fetchStatistics();
              setCycleHistory([]);
            }}
            className="btn btn-reset"
          >
            🔄 Reset
          </button>
          
          <div className="speed-control">
            <label>Speed: </label>
            <input 
              type="range" 
              min="100" 
              max="2000" 
              value={cycleSpeed}
              onChange={(e) => setCycleSpeed(Number(e.target.value))}
              disabled={!isRunning}
            />
            <span>{3000 - cycleSpeed} cycles/min</span>
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="main-grid">
          <div className="map-section">
            {worldState && <WorldMap worldState={worldState} onSelectRegion={setSelectedRegion} />}
          </div>

          <div className="right-panel">
            <div className="stats-section">
              {statistics && <StatisticsPanel statistics={statistics} />}
            </div>

            <div className="trade-section">
              {worldState && <TradeNetwork tradeData={worldState.trade_network} />}
            </div>
          </div>

          <div className="bottom-panel">
            <div className="region-details">
              {selectedRegion && worldState && (
                <RegionDashboard 
                  region={worldState.regions[selectedRegion]}
                  regionId={selectedRegion}
                />
              )}
            </div>

            <div className="events-section">
              {worldState && <EventLog events={worldState.events} />}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
