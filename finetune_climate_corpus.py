import os
from transformers import AutoModelForMaskedLM, EarlyStoppingCallback, TrainingArguments, Trainer
from torch import cuda
from datasets import Dataset
from transformers import AutoTokenizer

def load_text_files(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

data_path = "dataset\\fr"
texts = load_text_files(data_path)

dataset = Dataset.from_dict({"text": texts})
print(dataset)
print(dataset[1])


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="longest")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(tokenized_dataset)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)  # 90% train, 10% validation
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

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

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,                  
    eval_dataset=eval_dataset,                    
    tokenizer=tokenizer,                          
    data_collator=data_collator,                   
    callbacks=[early_stopping],                   
)

trainer.train()

output_dir = "./climate_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

