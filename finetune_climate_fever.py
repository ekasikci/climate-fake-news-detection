from datasets import Dataset
import json
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import cuda

data_file = 'dataset\\fr_climate-fever-dataset-r1_period_maj_opus-mt-tc-big-en-fr_v2-unicode.jsonl'  

claims = []
labels = []

label_mapping = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT_ENOUGH_INFO": "UNDECIDED",
    "DISPUTED": "UNDECIDED"
}

with open(data_file, 'r', encoding='utf-8') as file:
    for line in file:
        example = json.loads(line)
        
        claims.append(example['claim'])
        labels.append(label_mapping[example['claim_label']])
        

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


dataset = Dataset.from_dict({
    'claim': claims,
    'label': encoded_labels
})

saved_model_path = "./climate_model" 

model = AutoModelForSequenceClassification.from_pretrained(saved_model_path, num_labels=len(label_encoder.classes_))
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

print("Model and tokenizer loaded successfully.")


def tokenize_function(examples):
    return tokenizer(examples['claim'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)  # 90% train, 10% validation
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

print('train_dataset :', train_dataset)
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=100, 
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_strategy="epoch",
)

print(f"Using device: {cuda.get_device_name(0) if cuda.is_available() else 'cpu'}")

early_stopping = EarlyStoppingCallback(early_stopping_patience=6)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping],
)

trainer.train()

model_save_path = "./climate_fever_model" 
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved to {model_save_path}")
