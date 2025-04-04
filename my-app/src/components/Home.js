// src/components/Home.js
import React from 'react';
import './Home.css';

function Home() {
  return (
    <div className="home">
      <h2>Welcome to CyberVision AI</h2>
      <p>Your AI-powered assistant for understanding cyber threats and making data-driven security decisions.</p>

      <div className="content-box">
        <section className="intro-section">
          <h3>About CyberVision AI</h3>
          <p>
            CyberVision AI is a cybersecurity platform designed to assist security professionals, analysts, and researchers in identifying and understanding cyber threats using artificial intelligence. By leveraging data-driven models, CyberVision AI empowers users to stay one step ahead of attackers.
          </p>
        </section>

        <section className="analysis-guide">
          <h3>How to Use CyberVision AI for Analysis</h3>
          <ol>
            <li><strong>Enter</strong> a description of a suspicious activity or threat behavior in the input box.</li>
            <li><strong>Select</strong> your desired analysis type—whether you want insights on tactics, techniques, or general threat patterns.</li>
            <li><strong>Click “Analyze”</strong> to process your input. CyberVision AI will analyze the text and provide insights.</li>
            <li><strong>Review</strong> the results and explore additional learning resources for deeper insights.</li>
          </ol>
        </section>

        <section className="ttp-section">
          <h3>Understanding Techniques, Tactics, and Procedures (TTPs)</h3>
          <div className="ttp-box">
            <h4>Tactics</h4>
            <p>Tactics refer to the high-level goals of an adversary, such as gaining access or escalating privileges.</p>
            
            <h4>Techniques</h4>
            <p>Techniques describe the specific actions that adversaries use to accomplish their tactics.</p>
            
            <h4>Procedures</h4>
            <p>Procedures are unique sequences of actions that adversaries take to carry out techniques in real attacks.</p>
          </div>
        </section>

        <section className="learning-resources">
          <h3>Additional Learning Resources</h3>
          <ul>
            <li><a href="https://www.mitre.org/" target="_blank" rel="noopener noreferrer">MITRE ATT&CK Framework</a></li>
            <li><a href="https://www.cybrary.it/" target="_blank" rel="noopener noreferrer">Cybrary</a></li>
            <li><a href="https://owasp.org/" target="_blank" rel="noopener noreferrer">OWASP Foundation</a></li>
            <li><a href="https://threatpost.com/" target="_blank" rel="noopener noreferrer">ThreatPost</a></li>
          </ul>
        </section>
      </div>
    </div>
  );
}

export default Home;
