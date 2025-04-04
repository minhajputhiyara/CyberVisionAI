import React, { useState } from 'react';
import './Analyze.css';
import axios from 'axios';

const FormattedAnalysis = ({ description }) => {
  const parseAnalysis = (text) => {
    const sections = [];
    let currentSection = { title: '', content: [] };
    
    const lines = text.split('\n').map(line => line.trim()).filter(line => line);
    
    lines.forEach(line => {
      if (line.match(/^\d+\.\s+[A-Za-z\s]+:/)) {
        if (currentSection.title) {
          sections.push({ ...currentSection });
        }
        currentSection = {
          title: line,
          content: []
        };
      } else if (line.startsWith('-')) {
        currentSection.content.push({
          type: 'bullet',
          text: line.substring(1).trim()
        });
      } else {
        currentSection.content.push({
          type: 'text',
          text: line
        });
      }
    });
    
    if (currentSection.title) {
      sections.push(currentSection);
    }
    
    return sections;
  };

  const sections = parseAnalysis(description);

  return (
    <div className="analysis-container">
      {sections.map((section, idx) => (
        <div key={idx} className="analysis-section">
          <h3 className="analysis-title">{section.title}</h3>
          <div className="analysis-content">
            {section.content.map((content, contentIdx) => (
              content.type === 'bullet' ? (
                <div key={contentIdx} className="bullet-point">
                  <span className="bullet">•</span>
                  <p>{content.text}</p>
                </div>
              ) : (
                <p key={contentIdx} className="text-content">{content.text}</p>
              )
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

function Analyze() {
  const [inputText, setInputText] = useState("");
  const [techniquePrediction, setTechniquePrediction] = useState(null);
  const [tacticsPrediction, setTacticsPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!inputText) {
      alert("Please enter some text for analysis.");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', { text: inputText });
      const { technique, tactics } = response.data;
      setTechniquePrediction(technique);
      setTacticsPrediction(tactics);
    } catch (error) {
      alert("Error processing the request.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-container">
      <div className="input-section">
        <textarea
          placeholder="Enter text for analysis..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="analysis-input"
        />
        <button 
          onClick={handleAnalyze} 
          disabled={loading}
          className="analyze-button"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {(techniquePrediction || tacticsPrediction) && (
        <div className="results-container">
          {techniquePrediction && (
            <div className="prediction-card technique-card">
              <div className="card-header">
                <h2>Technique Analysis</h2>
                <div className="technique-id">ID: {techniquePrediction.id}</div>
              </div>
              {techniquePrediction.description && (
                <FormattedAnalysis description={techniquePrediction.description} />
              )}
            </div>
          )}

          {tacticsPrediction && (
            <div className="prediction-card tactics-card">
              <div className="card-header">
                <h2>Tactics Analysis</h2>
                <div className="tactics-id">ID: {tacticsPrediction.id}</div>
              </div>
              <div className="tactics-content">
                <h3>{tacticsPrediction.name}</h3>
                <p>{tacticsPrediction.description}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Analyze;