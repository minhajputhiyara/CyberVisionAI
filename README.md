# Cyber Threat Analysis and Prevention System

## Overview
This project aims to predict the techniques, tactics, and procedures (TTPs) of attackers by analyzing logs. Cyberattacks can be effectively analyzed and prevented using our automated and intelligent system, which understands the TTPs of cyber threats.

### Key Features
- **Automated Threat Analysis**: Predicts TTPs of attackers using advanced machine learning techniques.
- **Explainability**: Utilizes SHAP (SHapley Additive exPlanations) for model prediction explainability.
- **Context Understanding**: Employs BERT for better context understanding and classification.
- **Recommendations and Prevention**: Leverages LLaMA LLM to provide recommendations and prevention methods, ensuring user safety from threats.

---

## Technologies Used
- **SHAP**: For explainability of model predictions.
- **BERT**: For contextual understanding and classification.
- **LLaMA LLM**: To generate recommendations and prevention strategies.
- **FastAPI**: Backend framework for the application.
- **React**: Frontend framework for the application.

---

## Folder Structure
The project is organized into the following folders:

1. **`tram`**: Contains the dataset and related files.
2. **`crawler`**: Includes the crawler code for data collection and a notebook with the crawler implementation.
3. **`model`**: Contains the model training notebook and the trained model.
4. **`backend`**: FastAPI backend for the application.
5. **`my_app`**: React frontend for the application.

---

## How It Works
1. **Log Analysis**: The system analyzes logs to identify potential cyber threats.
2. **TTP Prediction**: Predicts the techniques, tactics, and procedures of attackers using a BERT-based model.
3. **Explainability**: SHAP is used to explain the model's predictions, providing insights into the decision-making process.
4. **Recommendations**: LLaMA LLM generates actionable recommendations and prevention methods to mitigate threats.

---

## Getting Started
### Prerequisites
- Python 3.8+
- Node.js (for React frontend)
- FastAPI (for backend)
- Required Python libraries (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MINI_PROJECT
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
4. Start the frontend:
   ```bash
   cd my_app
   npm install
   npm start
   ```

---

## Future Enhancements
- Integration with real-time log monitoring systems.
- Support for additional datasets and threat intelligence feeds.
- Enhanced visualization of SHAP explanations.

---

## Credits
This project was developed to provide an intelligent and automated solution for analyzing and preventing cyber threats.