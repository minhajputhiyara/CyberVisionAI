from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import random
from tactic_model import AttackPredictor
from technique_model import SecurityAnalyzer
import asyncio
import json
from fastapi.responses import StreamingResponse

app = FastAPI()

# Allow CORS from React (typically running on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

d1 = {
    "TA0001": {
        "name": "Initial Access",
        "description": "This is the first stage of an attack where the attacker gains a foothold in your system. They often do this through methods like phishing or exploiting vulnerabilities."
    },
    "TA0002": {
        "name": "Execution",
        "description": "Once inside, the attacker tries to execute malicious code on your system, running programs to carry out their attack."
    },
    "TA0003": {
        "name": "Persistence",
        "description": "The attacker ensures they can maintain access to your system, even if you try to remove them, by installing backdoors or setting up ways to reconnect."
    },
    "TA0004": {
        "name": "Privilege Escalation",
        "description": "The attacker tries to gain higher-level access to your system, giving them more control and allowing them to perform more dangerous actions."
    },
    "TA0005": {
        "name": "Defense Evasion",
        "description": "The attacker works to hide their presence and avoid being detected by security systems, making it harder for you to remove them."
    },
    "TA0006": {
        "name": "Credential Access",
        "description": "The attacker tries to steal usernames, passwords, or other credentials that can help them access more parts of the system or other systems."
    },
    "TA0007": {
        "name": "Discovery",
        "description": "The attacker tries to learn more about your system and network so they can plan their next moves."
    },
    "TA0008": {
        "name": "Lateral Movement",
        "description": "The attacker tries to move across your network, looking for more valuable information or systems to attack."
    },
    "TA0009": {
        "name": "Collection",
        "description": "The attacker gathers data from your system, such as sensitive information or intellectual property, to use it later."
    },
    "TA0010": {
        "name": "Exfiltration",
        "description": "The attacker transfers the collected data from your system to a location under their control, often for malicious use."
    },
    "TA0011": {
        "name": "Impact",
        "description": "The attacker aims to damage your system, destroy data, or disrupt your operations, often to cause financial or reputational harm."
    },
    "TA0012": {
        "name": "Command and Control",
        "description": "The attacker establishes communication with the system they've infected, allowing them to send commands and control the system remotely."
    },
    "TA0013": {
        "name": "Resource Development",
        "description": "The attacker sets up resources or infrastructure that can be used later in the attack, like creating fake accounts or gathering tools."
    },
    "TA0014": {
        "name": "Reconnaissance",
        "description": "The attacker gathers information about your systems and defenses before launching an attack, often using techniques like scanning networks or gathering publicly available data."
    },
    "TA0040": {
        "name": "Exploitation for Privilege Escalation",
        "description": "The attacker exploits vulnerabilities or weaknesses in the system to escalate their privileges and gain higher-level access."
    },
    "TA0042": {
        "name": "Resource Hijacking",
        "description": "The attacker compromises or hijacks system resources, such as CPUs, bandwidth, or storage, often for the purpose of cryptocurrency mining or launching further attacks."
    },
    "TA0043": {
        "name": "Data Destruction",
        "description": "The attacker destroys or corrupts data within the system, either as part of a broader attack strategy or to cause direct harm to the victim organization."
    }
}

d2 = {
    "technique": {
        "name": "technique not available now",
        "description": "this model is under training"
    }
}

# tactic prediction model
def tactic_predict(text):
    pred_tactic = AttackPredictor(
        model_path='tactic_model/best_model.pth',
        tactic_encoder_path='tactic_model/tactic_encoder.pkl',
        technique_encoder_path='tactic_model/technique_encoder.pkl'
    )
    prediction = pred_tactic.predict(text)
    return prediction['tactic']

# technique prediction model
def technique_predict(text):
    """Main function to run the security analyzer."""
    try:
        # Initialize analyzer
        analyzer = SecurityAnalyzer(
            model_path="model_seed_2455_best.pt",
            groq_api_key="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"
        )

        # Process and analyze
        processed_text = analyzer.preprocessor.preprocess(text)
        result = analyzer.predict(processed_text)
        groq_analysis = analyzer.get_groq_analysis(result)

        return result, groq_analysis

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

async def stream_analysis(text, groq_analysis):
    """Stream the analysis text word by word."""
    # Split sections by newline
    sections = groq_analysis.split('\n')
    
    for section in sections:
        # Split each section into words
        words = section.strip().split()
        
        # Stream each word
        for i, word in enumerate(words):
            yield f"{word}"
            if i < len(words) - 1:
                yield " "  # Add space between words
            await asyncio.sleep(0.05)  # 50ms delay between words
        
        # Add newline between sections
        if sections.index(section) < len(sections) - 1:
            yield "\n"

async def generate_stream_response(text_input):
    """Generate the streaming response."""
    text = text_input.text
    result, groq_analysis = technique_predict(text)
    tactics = tactic_predict(text)
    technique = result.predicted_label
    tactic_info = d1.get(tactics)

    # Send initial metadata as JSON
    initial_data = {
        'tactics': {
            'id': tactics,
            'name': tactic_info['name'],
            'description': tactic_info['description']
        },
        'technique': {
            'id': technique,
            'name': {'shap':result.word_attributions,'confidence':result.confidence,'topPredictions':result.top_predictions},
        }
    }
    yield (json.dumps(initial_data) + '\n').encode('utf-8')

    # Stream the analysis text
    async for chunk in stream_analysis(text, groq_analysis):
        yield chunk.encode('utf-8')

@app.post("/predict/stream")
async def stream_prediction(text_input: TextInput):
    """Endpoint for streaming predictions."""
    return StreamingResponse(
        generate_stream_response(text_input),
        media_type='text/plain',
    )

# Keep the original endpoint for compatibility
@app.post("/predict/")
async def get_first_word(text_input: TextInput):
    print(text_input)
    text = str(text_input)
    result, groq_analysis = technique_predict(text)
    tactics = tactic_predict(text)
    technique = result.predicted_label
    analyze_report = groq_analysis
    tactic_info = d1.get(tactics)
    print(analyze_report)
    print("confidence")
    print(result.confidence)

    return {
        'tactics': {
            'id': tactics,
            'name': tactic_info['name'],
            'description': tactic_info['description']
        },
        'technique': {
            'id': technique,
            'name': result.confidence,
            'description': analyze_report
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)