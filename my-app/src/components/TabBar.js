// src/components/TabBar.js
import React from 'react';
import './TabBar.css';

function TabBar({ activeTab, setActiveTab }) {
  return (
    <div className="tab-bar">
      <span 
        onClick={() => setActiveTab("home")} 
        className={`tab-bar-item ${activeTab === "home" ? "active" : ""}`}
      >
        Home
      </span>
      <span 
        onClick={() => setActiveTab("analyze")} 
        className={`tab-bar-item ${activeTab === "analyze" ? "active" : ""}`}
      >
        Analyze
      </span>
      <span 
        onClick={() => setActiveTab("resources")} 
        className={`tab-bar-item ${activeTab === "resources" ? "active" : ""}`}
      >
        Learning Resources
      </span>
      <span 
        onClick={() => setActiveTab("insights")} 
        className={`tab-bar-item ${activeTab === "insights" ? "active" : ""}`}
      >
        Insights
      </span>
    </div>
  );
}

export default TabBar;
