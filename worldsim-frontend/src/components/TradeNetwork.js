import React, { useState, useEffect } from 'react';
import './TradeNetwork.css';

function TradeNetwork({ tradeData }) {
  const [connections, setConnections] = useState([]);

  useEffect(() => {
    if (tradeData && tradeData.edges) {
      setConnections(tradeData.edges);
    }
  }, [tradeData]);

  const getRegionName = (regionId) => {
    const names = {
      'region_0': 'Northern Plains',
      'region_1': 'Forest Kingdom',
      'region_2': 'Coastal Republic',
      'region_3': 'Desert Alliance',
      'region_4': 'Mountain Federation',
      'region_5': 'Valley States'
    };
    return names[regionId] || regionId;
  };

  return (
    <div className="trade-network">
      <h3>🤝 Trade Network</h3>
      
      <div className="network-stats">
        <div className="stat-item">
          <span>Active Routes</span>
          <span className="value">{connections.length}</span>
        </div>
      </div>

      <div className="trade-list">
        <h4>Trade Connections</h4>
        {connections.length === 0 ? (
          <p className="no-trades">No active trade routes</p>
        ) : (
          connections.map((connection, idx) => (
            <div key={idx} className="trade-item">
              <div className="trade-info">
                <span className="region">{getRegionName(connection.source)}</span>
                <span className="arrow">↔</span>
                <span className="region">{getRegionName(connection.target)}</span>
              </div>
              <div className="trade-strength">
                <div className="strength-bar">
                  <div 
                    className="strength-fill"
                    style={{ width: `${connection.weight * 100}%` }}
                  />
                </div>
                <span>{(connection.weight * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default TradeNetwork;
