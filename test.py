import json
import joblib 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
from sklearn.metrics import classification_report


saved_model_path = "./climate_fever_model"  
model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

label_encoder = joblib.load('label_encoder.pkl')  
label_mapping = label_encoder.classes_.tolist() 
label_to_index = {label: idx for idx, label in enumerate(label_mapping)}

label_map = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT_ENOUGH_INFO": "UNDECIDED",
    "DISPUTED": "UNDECIDED"
}

def predict_label_with_probs(claim):
    features = tokenizer(
        [claim], 
        padding='longest', 
        truncation=True, 
        return_tensors="pt", 
    )
    
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        print(scores)
        probs = F.softmax(scores, dim=-1)
        predicted_label_idx = probs.argmax(dim=1).item()
        predicted_label = label_mapping[predicted_label_idx] 
    
    return predicted_label

dataset_path = "dataset\\fr_climate-fever-dataset-r1_period_maj_opus-mt-tc-big-en-fr_v2-unicode.jsonl"
true_labels = []
predicted_labels = []

with open(dataset_path, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        claim = data["claim"]
        claim_label = data["claim_label"]
        
        if claim_label == 'DISPUTED':
            continue
        
        mapped_claim_label = label_map.get(claim_label, "UNDECIDED")
        
        true_label_idx = label_encoder.transform([mapped_claim_label])[0]
        
        predicted_label = predict_label_with_probs(claim)
        predicted_label_idx = label_to_index[predicted_label]
        
        true_labels.append(true_label_idx)
        predicted_labels.append(predicted_label_idx)
        
        

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("True label mapping:", label_mapping)
print("Predicted label mapping:", label_mapping)


print(f"Label classes (order matters!): {label_mapping}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nClass-wise performance:")
print(classification_report(true_labels, predicted_labels, target_names=label_mapping))