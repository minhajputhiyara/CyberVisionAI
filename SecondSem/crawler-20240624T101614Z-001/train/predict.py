import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
import numpy as np
import os

class BERTClass(torch.nn.Module):
    def __init__(self, pretrained_model_name: str, num_classes: int = None, dropout: float = 0.5):
        super().__init__()
        config = BertConfig.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name, config=config).base_model
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class ModelPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = 256
        
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.label_map = checkpoint['label_map']
        
        num_classes = len(self.label_map)
        self.model = BERTClass("bert-base-uncased", num_classes)
        
        if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully with accuracy: {checkpoint['accuracy']:.4f}")
        print(f"Label mapping: {self.label_map}")

    def preprocess_text(self, text):
        text = str(text)
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
        }

    def predict(self, text):
        inputs = self.preprocess_text(text)
        ids = inputs['ids'].to(self.device)
        mask = inputs['mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(ids, mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        predicted_label = self.label_map[prediction.item()]
        confidence = confidence.item()
        
        top_probs, top_indices = torch.topk(probabilities, min(3, len(self.label_map)))
        top_predictions = [
            {
                'label': self.label_map[idx.item()],
                'confidence': prob.item()
            }
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]
        
        return {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'top_predictions': top_predictions
        }

    def predict_batch(self, texts):
        predictions = []
        for text in texts:
            prediction = self.predict(text)
            predictions.append(prediction)
        return predictions

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "model_seed_2455_best.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predictor = ModelPredictor(model_path, device)
    
    # Read input from terminal
    text = input("Enter the text to analyze: ")
    result = predictor.predict(text)
    
    print("\nPrediction Results:")
    print(f"Text: {text}")
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop 3 Predictions:")
    for pred in result['top_predictions']:
        print(f"Label: {pred['label']}, Confidence: {pred['confidence']:.4f}")
