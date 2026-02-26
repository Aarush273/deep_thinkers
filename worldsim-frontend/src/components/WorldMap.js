import React, { useState } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts';
import './WorldMap.css';

function WorldMap({ worldState, onSelectRegion }) {
  const [hoveredRegion, setHoveredRegion] = useState(null);

  const getRegionColor = (region) => {
    if (region.resources.food < 100 || region.resources.energy < 100) {
      return '#ff4444'; // Critical
    }
    if (region.stability < 0.5) {
      return '#ffaa00'; // Warning
    }
    return '#44aa44'; // Healthy
  };

  const resourceData = (region) => [
    { name: 'Water', value: region.resources.water, fill: '#3399ff' },
    { name: 'Food', value: region.resources.food, fill: '#ff9933' },
    { name: 'Energy', value: region.resources.energy, fill: '#ffff33' },
  ];

  return (
    <div className="world-map">
      <h2>🗺️ World Overview</h2>
      <div className="regions-grid">
        {Object.entries(worldState.regions).map(([regionId, region]) => (
          <div
            key={regionId}
            className={`region-card ${hoveredRegion === regionId ? 'hovered' : ''}`}
            onClick={() => onSelectRegion(regionId)}
            onMouseEnter={() => setHoveredRegion(regionId)}
            onMouseLeave={() => setHoveredRegion(null)}
            style={{ borderColor: getRegionColor(region) }}
          >
            <div className="region-header">
              <h4>{region.name}</h4>
              <span className="pop-badge">👥 {region.population}</span>
            </div>

            <div className="region-resources">
              <ResponsiveContainer width="100%" height={120}>
                <PieChart>
                  <Pie
                    data={resourceData(region)}
                    cx="50%"
                    cy="50%"
                    innerRadius={20}
                    outerRadius={45}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {resourceData(region).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="region-stats">
              <div><span>Stability:</span> {(region.stability * 100).toFixed(0)}%</div>
              <div><span>Dev Level:</span> {(region.development_level * 100).toFixed(0)}%</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default WorldMap;
