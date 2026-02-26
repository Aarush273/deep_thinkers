import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './RegionDashboard.css';

const API_BASE = 'http://localhost:8000';

function RegionDashboard({ region, regionId }) {
  const [agentStats, setAgentStats] = useState(null);

  useEffect(() => {
    const fetchAgentStats = async () => {
      try {
        const response = await axios.get(`${API_BASE}/regions/${regionId}`);
        setAgentStats(response.data.agent_stats);
      } catch (error) {
        console.error('Failed to fetch agent stats:', error);
      }
    };

    fetchAgentStats();
  }, [regionId]);

  if (!region) return <div>Select a region to view details</div>;

  return (
    <div className="region-dashboard">
      <h3>📍 {region.name} - Detailed Analysis</h3>

      <div className="dashboard-grid">
        <div className="dashboard-section">
          <h4>Resources</h4>
          <div className="resource-bars">
            {['water', 'food', 'energy', 'land'].map((resource) => (
              <div key={resource} className="resource-bar-item">
                <label>{resource.charAt(0).toUpperCase() + resource.slice(1)}</label>
                <div className="bar">
                  <div
                    className="fill"
                    style={{
                      width: `${(region.resources[resource] / (resource === 'land' ? 1000 : 2000)) * 100}%`,
                      backgroundColor: resource === 'water' ? '#3399ff' : 
                                     resource === 'food' ? '#ff9933' :
                                     resource === 'energy' ? '#ffff33' : '#99ff99'
                    }}
                  />
                </div>
                <span>{region.resources[resource].toFixed(0)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="dashboard-section">
          <h4>Demographics</h4>
          <div className="stat-list">
            <div><span>Population:</span> {region.population}</div>
            <div><span>Growth Rate:</span> {(region.growth_rate * 100).toFixed(2)}%</div>
            <div><span>Development Level:</span> {(region.development_level * 100).toFixed(1)}%</div>
            <div><span>Stability:</span> {(region.stability * 100).toFixed(1)}%</div>
          </div>
        </div>

        <div className="dashboard-section">
          <h4>Environment</h4>
          <div className="stat-list">
            <div><span>Temperature:</span> {region.temperature.toFixed(1)}°C</div>
            <div><span>Rainfall:</span> {region.rainfall.toFixed(1)}mm/month</div>
            <div><span>Disaster Risk:</span> {(region.disaster_risk * 100).toFixed(2)}%</div>
            <div><span>Trade Partners:</span> {Object.keys(region.trade_partners).length}</div>
          </div>
        </div>

        {agentStats && (
          <div className="dashboard-section">
            <h4>Agent Intelligence</h4>
            <div className="stat-list">
              <div><span>Actions Taken:</span> {agentStats.actions_taken}</div>
              <div><span>Avg Reward:</span> {agentStats.avg_reward.toFixed(2)}</div>
              <div><span>Learning Progress:</span> {(agentStats.epsilon * 100).toFixed(1)}%</div>
              <div className="recent-actions">
                <span>Last Actions:</span>
                <div className="action-tags">
                  {agentStats.recent_actions.map((action, idx) => (
                    <span key={idx} className="action-tag">{action}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RegionDashboard;
