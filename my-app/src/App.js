// App.js
import React, { useState } from 'react';
import './App.css';
import TabBar from './components/TabBar';
import Home from './components/Home';
import Analyze from './components/Analyze';
import LearningResources from './components/LearningResources';
import Insights from './components/Insights';

function App() {
  const [activeTab, setActiveTab] = useState("home");

  return (
    <div className="app-container">
      <header className="header">CyberVisi👁️n AI</header>

      {/* Tab Navigation */}
      <TabBar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Render Active Tab Content */}
      <div className="content-section">
        {activeTab === "home" && <Home />}
        {activeTab === "analyze" && <Analyze />}
        {activeTab === "resources" && <LearningResources />}
        {activeTab === "insights" && <Insights />}
      </div>
    </div>
  );
}

export default App;
