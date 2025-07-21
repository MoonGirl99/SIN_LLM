import sys
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

if len(sys.argv) != 2:
    print('usage python species_llama_token_clf_standard.py model_size')
    sys.exit()

model_size = sys.argv[1].lower()
print(f'handling species NER with model size {model_size}')

# Configuration
epochs = 5
batch_size = 8
learning_rate = 1e-4
max_length = 256
lora_r = 12

# Model selection
if model_size == '3b':
    model_id = 'meta-llama/Llama-3.2-3B-Instruct'  # or 'unsloth/Llama-3.2-3B-Instruct' 
else:
    raise NotImplementedError

# Load dataset
print("Loading dataset...")
ds = load_dataset("spyysalo/species_800")
print(f"Dataset loaded: {ds}")

# Get the label list from the dataset
label_list = ds["train"].features["ner_tags"].feature.names
print(f"Labels: {label_list}")
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}

# Initialize tokenizer and model
print(f"Loading tokenizer from {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

print(f"Loading model from {model_id}...")
# Use the standard token classification model
model = AutoModelForTokenClassification.from_pretrained(
    model_id, 
    num_labels=len(label2id), 
    id2label=id2label, 
    label2id=label2id
)

# Apply LoRA for parameter-efficient fine-tuning
print("Applying LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, 
    inference_mode=False, 
    r=lora_r, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load evaluation metric
seqeval = evaluate.load("seqeval")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        is_split_into_words=True, 
        padding='longest', 
        max_length=max_length, 
        truncation=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  #
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Process the dataset
print("Processing dataset...")
tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Training configuration
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="species_finedtuned_model_less_epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    gradient_accumulation_steps=4,  
    fp16=True,  
)

# Initialize trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()
print("Training complete!")

# Save the model
print("Saving model...")
trainer.save_model("species_finedtuned_model")
print("Model saved successfully!")