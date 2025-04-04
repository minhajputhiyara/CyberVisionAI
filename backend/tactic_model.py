import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

class AttackPredictor:
    def __init__(self, model_path='best_model.pth', 
                 tactic_encoder_path='tactic_encoder.pkl',
                 technique_encoder_path='technique_encoder.pkl'):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load label encoders
        with open(tactic_encoder_path, 'rb') as f:
            self.tactic_encoder = pickle.load(f)
        with open(technique_encoder_path, 'rb') as f:
            self.technique_encoder = pickle.load(f)
        
        # Initialize and load model
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(self.tactic_encoder.classes_)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, max_length=128):
        """
        Predict tactic for a given text.
        
        Args:
            text (str): Input text to classify
            max_length (int): Maximum sequence length
            
        Returns:
            dict: Dictionary containing predicted tactic and confidence score
        """
        # Tokenize input text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert predicted class back to label
        predicted_tactic = self.tactic_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'tactic': predicted_tactic,
            'confidence': confidence
        }

    def predict_batch(self, texts, max_length=128):
        """
        Predict tactics for a batch of texts.
        
        Args:
            texts (list): List of input texts to classify
            max_length (int): Maximum sequence length
            
        Returns:
            list: List of dictionaries containing predictions and confidence scores
        """
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = [probabilities[i][predicted_classes[i]].item() for i in range(len(texts))]
        
        # Convert predicted classes back to labels
        predicted_tactics = self.tactic_encoder.inverse_transform(predicted_classes)
        
        # Create list of predictions
        predictions = [
            {'tactic': tactic, 'confidence': confidence}
            for tactic, confidence in zip(predicted_tactics, confidences)
        ]
        
        return predictions

# Example usage
def main():
    # Initialize predictor
    predictor = AttackPredictor()
    
    # Single prediction example
    text = "An attacker used a phishing email to steal credentials"
    prediction = predictor.predict(text)
    print(f"\nSingle prediction:")
    print(f"Text: {text}")
    print(f"Predicted tactic: {prediction['tactic']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    # Batch prediction example
    texts = [
        "An attacker used a phishing email to steal credentials",
        "The malware established persistence by modifying the registry",
        "Data was exfiltrated through an encrypted channel"
    ]
    predictions = predictor.predict_batch(texts)
    print(f"\nBatch predictions:")
    for text, pred in zip(texts, predictions):
        print(f"\nText: {text}")
        print(f"Predicted tactic: {pred['tactic']}")
        print(f"Confidence: {pred['confidence']:.2%}")

if __name__ == "__main__":
    main()