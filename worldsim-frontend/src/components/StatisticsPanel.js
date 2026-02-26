import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './StatisticsPanel.css';

function StatisticsPanel({ statistics }) {
  const data = [
    { name: 'Avg Stability', value: (statistics.avg_stability * 100).toFixed(1) },
    { name: 'Avg Development', value: (statistics.avg_development * 100).toFixed(1) },
  ];

  return (
    <div className="stats-panel">
      <h3>📊 Simulation Statistics</h3>
      
      <div className="stat-grid">
        <div className="stat-item">
          <span className="stat-label">Cycle</span>
          <span className="stat-value">{statistics.cycle}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Population</span>
          <span className="stat-value">{statistics.total_population}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Active Regions</span>
          <span className="stat-value">{statistics.active_regions}/{statistics.active_regions + statistics.collapsed_regions}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Trade Routes</span>
          <span className="stat-value">{statistics.trade_connections}</span>
        </div>
      </div>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default StatisticsPanel;
