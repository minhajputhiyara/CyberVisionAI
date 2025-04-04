import React, { useState, useRef, useEffect } from 'react';
import './Analyze.css';

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
  const [streamingContent, setStreamingContent] = useState("");
  const [loading, setLoading] = useState(false);
  const readerRef = useRef(null);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (readerRef.current) {
        readerRef.current.cancel();
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const handleStreamingAnalyze = async () => {
    if (!inputText) {
      alert("Please enter some text for analysis.");
      return;
    }

    setLoading(true);
    setStreamingContent("");
    setTechniquePrediction(null);
    setTacticsPrediction(null);

    // Create new AbortController
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('http://127.0.0.1:8000/predict/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
        signal: abortControllerRef.current.signal
      });

      const reader = response.body.getReader();
      readerRef.current = reader;
      
      let receivedMetadata = false;
      let accumulatedContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        
        if (!receivedMetadata) {
          // First line contains JSON metadata
          const newlineIndex = text.indexOf('\n');
          if (newlineIndex !== -1) {
            const metadataStr = text.slice(0, newlineIndex);
            const metadata = JSON.parse(metadataStr);
            setTacticsPrediction(metadata.tactics);
            setTechniquePrediction({
              ...metadata.technique,
              description: "" // Will be filled by streaming content
            });
            receivedMetadata = true;
            accumulatedContent = text.slice(newlineIndex + 1);
          }
        } else {
          accumulatedContent += text;
        }

        // Update technique prediction with accumulated content
        setTechniquePrediction(prev => ({
          ...prev,
          description: accumulatedContent
        }));
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        alert("Error processing the request.");
      }
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
          onClick={handleStreamingAnalyze} 
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
                <div className="technique-id">Technique: {techniquePrediction.id}</div>
                <p>Confidence: {techniquePrediction.name.confidence}</p>
              </div>
              <div>
                {/* Handle SHAP values */}
                <p>SHAP Values:</p>
          {Array.isArray(techniquePrediction.name.shap) && techniquePrediction.name.shap.length > 0 ? (
            <ul>
              {techniquePrediction.name.shap.map((shapItem, index) => (
                <li key={index}>
                  <strong>{shapItem.word}:</strong> {shapItem.attribution.toFixed(4)} {/* Display attribution with 4 decimal places */}
                </li>
              ))}
            </ul>
          ) : (
            <p>No SHAP values available.</p>
          )}
        </div>



        <div>
      <h3>Top Predictions:</h3>
      {Array.isArray(techniquePrediction.name.topPredictions) && techniquePrediction.name.topPredictions.length > 0 ? (
        <ul>
          {techniquePrediction.name.topPredictions.map((pred, index) => (
            <li key={index}>
              <strong>{pred.label}:</strong> Confidence: {pred.confidence.toFixed(4)} {/* Display confidence with 4 decimal places */}
            </li>
          ))}
        </ul>
      ) : (
        <p>No top predictions available.</p>
      )}
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