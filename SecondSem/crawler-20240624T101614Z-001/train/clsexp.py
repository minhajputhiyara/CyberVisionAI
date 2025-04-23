import torch
import pandas as pd
import json
import re
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
from groq import Groq
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Data class to store prediction results."""
    text: str
    predicted_label: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    word_attributions: List[Dict[str, Any]]

class TextPreprocessor:
    """Handles text preprocessing operations."""
    
    def __init__(self, min_word_length: int = 2):
        self.min_word_length = min_word_length
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why',
            'can', 'could', 'should', 'would', 'may', 'might', 'must', 'shall'
        }
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return ' '.join(text.split())
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords and short words from text."""
        words = text.split()
        words = [
            word for word in words 
            if word not in self.stopwords and len(word) >= self.min_word_length
        ]
        return ' '.join(words)
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline."""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

class BERTClassifier(torch.nn.Module):
    """BERT-based text classifier with custom architecture."""
    
    def __init__(
        self, 
        pretrained_model_name: str, 
        num_classes: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.config = BertConfig.from_pretrained(
            pretrained_model_name, 
            output_hidden_states=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name, 
            config=self.config
        ).base_model
        
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, num_classes)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass of the model."""
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        return self.classifier(pooler)

class SecurityAnalyzer:
    """Main class for security text analysis."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        groq_api_key: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = TextPreprocessor()
        self.model_path = Path(model_path)
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        
        self._initialize_model()
        logger.info(f"Model initialized successfully on {self.device}")
        
    def _initialize_model(self) -> None:
        """Initialize the BERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            checkpoint = torch.load(
                self.model_path, 
                map_location=torch.device(self.device)
            )
            
            # Convert label map if necessary
            self.label_map = {
                int(k) if isinstance(k, str) else k.item() if torch.is_tensor(k) else k: v
                for k, v in checkpoint['label_map'].items()
            }
            
            num_classes = len(self.label_map)
            logger.info(f"Loaded {num_classes} classes: {self.label_map}")
            
            self.model = BERTClassifier("bert-base-uncased", num_classes)
            
            # Handle DataParallel if necessary
            if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                self.model = torch.nn.DataParallel(self.model)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _get_label(self, index: Union[int, torch.Tensor]) -> str:
        """Safely get label from index."""
        if torch.is_tensor(index):
            index = index.item()
        return self.label_map.get(index, f"Unknown_{index}")

    def predict(self, text: str) -> PredictionResult:
        """Make predictions on input text."""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=256,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                output = self.model(**inputs)
            
            # Calculate probabilities
            probs = torch.softmax(output, dim=-1)
            
            # Get top predictions
            values, indices = probs[0].topk(min(5, len(self.label_map)))
            
            top_predictions = [
                {
                    'label': self._get_label(idx),
                    'confidence': float(val)
                }
                for val, idx in zip(values, indices)
            ]
            
            # Calculate word attributions
            words = text.split()
            word_attributions = [
                {"word": word, "attribution": float(probs[0].max())}
                for word in words
            ]
            
            return PredictionResult(
                text=text,
                predicted_label=top_predictions[0]['label'],
                confidence=top_predictions[0]['confidence'],
                top_predictions=top_predictions,
                word_attributions=word_attributions
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_groq_analysis(
        self, 
        result: PredictionResult
    ) -> Optional[str]:
        """Get detailed analysis from Groq API."""
        if not self.groq_client:
            logger.warning("Groq client not initialized - skipping analysis")
            return None
            
        try:
            # Format SHAP values and predictions
            shap_text = "\n".join(
                f"- {attr['word']}: {attr['attribution']:.4f}"
                for attr in sorted(
                    result.word_attributions,
                    key=lambda x: abs(x['attribution']),
                    reverse=True
                )[:10]
            )
            
            pred_text = "\n".join(
                f"- {pred['label']}: {pred['confidence']:.4f}"
                for pred in result.top_predictions
            )
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(
                result.text,
                shap_text,
                pred_text
            )
            
            # Get Groq completion
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert focused on threat detection and analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=2048
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting Groq analysis: {str(e)}")
            return None

    @staticmethod
    def _create_analysis_prompt(
        text: str,
        shap_text: str,
        pred_text: str
    ) -> str:
        """Create the analysis prompt for Groq."""
        return f"""
        Analyze the following security text classification results:

        Input Text: "{text}"

        Top SHAP Values (Most influential words):
        {shap_text}

        Model Predictions and Confidence Scores:
        {pred_text}

        Please provide:
        1. Threat Analysis
        2. Word Impact Analysis
        3. Attack Vector Analysis
        4. Recommendations
        """

def main():
    """Main function to run the security analyzer."""
    try:
        # Initialize analyzer
        analyzer = SecurityAnalyzer(
            model_path="model_seed_2455_best.pt",
            groq_api_key="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"
        )
        
        while True:
            # Get input
            print("\nEnter text to analyze (or 'quit' to exit):")
            text = input().strip()
            
            if text.lower() == 'quit':
                break
                
            if not text:
                print("Please enter some text to analyze.")
                continue
            
            # Process and analyze
            processed_text = analyzer.preprocessor.preprocess(text)
            result = analyzer.predict(processed_text)
            
            # Get Groq analysis if available
            groq_analysis = analyzer.get_groq_analysis(result)
            
            # Display results
            print("\n=== Analysis Results ===")
            print(f"Original Text: {text}")
            #print(f"Processed Text: {processed_text}")
            print(f"Predicted Label: {result.predicted_label}")
            print(f"Confidence: {result.confidence:.4f}")
            
            print("\nTop Contributing Words:")
            for attr in sorted(
                result.word_attributions,
                key=lambda x: abs(x['attribution']),
                reverse=True
            )[:10]:
                print(f"- {attr['word']}: {attr['attribution']:.4f}")
            
            if groq_analysis:
                print("\nDetailed Security Analysis:")
                print(groq_analysis)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
