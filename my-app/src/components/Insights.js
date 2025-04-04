// src/components/Insights.js
import React from 'react';
import './Insights.css';

function Insights() {
  return (
    <div className="insights-container">
      <h2 className="insights-heading">Insights & Analysis</h2>

      <div className="insights-content">
        <section className="insight-section">
          <h3>Top Security Trends</h3>
          <p>Stay informed about the latest cybersecurity trends and threats affecting your domain. Our insights provide up-to-date information on the evolving landscape of security risks and best practices.</p>
        </section>

        <section className="insight-section">
          <h3>Analysis Metrics</h3>
          <p>Analyze key metrics derived from recent analyses. Here’s where you’ll find the breakdown of prediction models, detection accuracy, and actionable insights based on your data.</p>
        </section>

        <section className="insight-section">
          <h3>Recommended Learning Resources</h3>
          <ul>
            <li><a href="https://example.com/learning-resource-1" target="_blank" rel="noopener noreferrer">Machine Learning in Cybersecurity</a></li>
            <li><a href="https://example.com/learning-resource-2" target="_blank" rel="noopener noreferrer">Advanced Threat Detection Techniques</a></li>
            <li><a href="https://example.com/learning-resource-3" target="_blank" rel="noopener noreferrer">Cybersecurity Analysis and Reporting</a></li>
          </ul>
        </section>
      </div>
    </div>
  );
}

export default Insights;
