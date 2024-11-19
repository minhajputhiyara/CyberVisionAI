// src/components/LearningResources.js
import React from 'react';
import './LearningResources.css';

function LearningResources() {
  return (
    <div className="resources-section">
      <h2>Learning Resources</h2>
      <div className="resource-card">
        <h3>Cybersecurity Basics</h3>
        <p>Learn about the fundamental concepts of cybersecurity, including encryption, authentication, and threat detection.</p>
        <a href="https://www.cybrary.it/course/intro-to-cyber-security/" target="_blank" rel="noopener noreferrer">Start Learning</a>
      </div>
      <div className="resource-card">
        <h3>Threat Intelligence</h3>
        <p>Understand how threat intelligence helps organizations stay ahead of potential threats and the key strategies to implement.</p>
        <a href="https://www.sans.org/cyber-security-courses/threat-intelligence/" target="_blank" rel="noopener noreferrer">Start Learning</a>
      </div>
      <div className="resource-card">
        <h3>AI in Cybersecurity</h3>
        <p>Explore how Artificial Intelligence is used to detect anomalies, predict attacks, and improve overall cybersecurity effectiveness.</p>
        <a href="https://www.ibm.com/security/artificial-intelligence" target="_blank" rel="noopener noreferrer">Start Learning</a>
      </div>
      <div className="resource-card">
        <h3>Understanding TTPs (Tactics, Techniques, Procedures)</h3>
        <p>Get a deep dive into TTPs, the behaviors and methodologies used by cyber adversaries, to better defend against attacks.</p>
        <a href="https://www.mitre.org/research/featured-research/mitre-att-andck-framework" target="_blank" rel="noopener noreferrer">Start Learning</a>
      </div>
      <div className="resource-card">
        <h3>Advanced Threat Hunting</h3>
        <p>Learn how threat hunters proactively search for threats inside an organization's network before they cause harm.</p>
        <a href="https://www.sans.org/cyber-security-courses/advanced-threat-hunting/" target="_blank" rel="noopener noreferrer">Start Learning</a>
      </div>
      <div className="resource-card">
        <h3>Incident Response and Management</h3>
        <p>Understand how to manage and respond to security incidents and how to build an effective incident response strategy.</p>
        <a href="https://www.coursera.org/learn/incident-response" target="_blank" rel="noopener noreferrer">Start Learning</a>
      </div>
    </div>
  );
}

export default LearningResources;
