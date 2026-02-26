import React from 'react';
import './EventLog.css';

function EventLog({ events }) {
  const getEventEmoji = (eventType) => {
    const emojiMap = {
      'drought': '🏜️',
      'flood': '🌊',
      'heatwave': '🔥',
      'coldsnap': '❄️',
      'harvest': '🌾',
      'plague': '😷'
    };
    return emojiMap[eventType] || '⚡';
  };

  const getEventColor = (eventType) => {
    const colorMap = {
      'drought': '#d4a574',
      'flood': '#4a90e2',
      'heatwave': '#ff6b35',
      'coldsnap': '#5b9bd5',
      'harvest': '#70ad47',
      'plague': '#c55a11'
    };
    return colorMap[eventType] || '#888';
  };

  return (
    <div className="event-log">
      <h3>⚡ Recent Events</h3>
      
      <div className="events-list">
        {events && events.length > 0 ? (
          events.map((event, idx) => (
            <div 
              key={idx} 
              className="event-item"
              style={{ borderLeftColor: getEventColor(event.type) }}
            >
              <div className="event-emoji">{getEventEmoji(event.type)}</div>
              <div className="event-details">
                <div className="event-title">
                  {event.type.charAt(0).toUpperCase() + event.type.slice(1)}
                </div>
                <div className="event-regions">
                  {event.affected_regions.map((region, i) => (
                    <span key={i} className="region-badge">
                      {region.replace('region_', 'R')}
                    </span>
                  ))}
                </div>
              </div>
              <div className="event-severity">
                <span className="severity-label">Severity</span>
                <span className="severity-value">{(event.severity * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))
        ) : (
          <p className="no-events">No events yet</p>
        )}
      </div>
    </div>
  );
}

export default EventLog;
