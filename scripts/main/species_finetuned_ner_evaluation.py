import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


def load_finetuned_model(model_path, model_id, num_labels=3):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label={0: "O", 1: "B", 2: "I"},
        label2id={"O": 0, "B": 1, "I": 2},
        local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def predict_ner(tokens, model, tokenizer, device):
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    word_ids = inputs.word_ids()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = logits.argmax(dim=-1)[0].cpu().numpy()
    id2label = {0: "O", 1: "B", 2: "I"}
    word_preds = []
    prev_word_id = None
    current_preds = []
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            if current_preds:
                word_preds.append(id2label[max(set(current_preds), key=current_preds.count)])
            current_preds = [predictions[i]]
            prev_word_id = word_id
        else:
            current_preds.append(predictions[i])
    if current_preds:
        word_preds.append(id2label[max(set(current_preds), key=current_preds.count)])
    return word_preds


def evaluate_model(model_path, model_id="meta-llama/Llama-3.2-3B-Instruct", num_samples=None, save_csv=True, csv_path="finetuned_species_ner_metrics.csv"):
    model, tokenizer, device = load_finetuned_model(model_path, model_id)
    dataset = load_dataset("spyysalo/species_800")["test"]
    if num_samples:
        dataset = dataset.select(range(num_samples))
    label_list = ["O", "B", "I"]
    y_true, y_pred = [], []
    print(f"Evaluating on {len(dataset)} samples...")
    for example in tqdm(dataset):
        tokens = example["tokens"]
        true_labels = [label_list[i] for i in example["ner_tags"]]
        pred_labels = predict_ner(tokens, model, tokenizer, device)
        min_len = min(len(true_labels), len(pred_labels))
        y_true.append(true_labels[:min_len])
        y_pred.append(pred_labels[:min_len])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print("\n--- Final Evaluation Metrics ---")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("\nDetailed Report:\n", report)
    if save_csv:
        pd.DataFrame([{
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }]).to_csv(csv_path, index=False)
    return {"precision": precision, "recall": recall, "f1_score": f1, "report": report} 