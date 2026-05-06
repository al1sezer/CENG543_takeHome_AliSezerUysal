import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging
import os
from seqeval.metrics import f1_score, classification_report
from data_utils import get_conll_dataset, tokenize_and_align_labels

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mute redundant library logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Constants
MODEL_NAME = "distilbert-base-cased"
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 30
PATIENCE = 3

def evaluate_transformer(model, dataloader, device, id2label):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Convert tags to strings, ignore masked tokens (-100)
            for i in range(len(labels)):
                true_tags = []
                pred_tags = []
                for label_id, pred_id in zip(labels[i], predictions[i]):
                    if label_id.item() != -100:
                        true_tags.append(id2label[label_id.item()])
                        pred_tags.append(id2label[pred_id.item()])
                
                if true_tags:
                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)
                    
    return f1_score(all_labels, all_preds), classification_report(all_labels, all_preds)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {MODEL_NAME}")

    dataset = get_conll_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tag structure
    labels = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    # Tokenize and align labels
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, MAX_LEN), 
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_datasets.set_format("torch")

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0
    patience_counter = 0

    logger.info(f"Fine-tuning started (Max Epoch: {EPOCHS}, Batch: {BATCH_SIZE})...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_f1, _ = evaluate_transformer(model, val_loader, device, id2label)
        logger.info(f">> [Epoch {epoch+1:02d}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        # Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_distilbert_ner.pt")
            logger.info("Best transformer model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered (Transformer).")
                break

    # Final Test
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load("best_distilbert_ner.pt"))
    test_f1, test_report = evaluate_transformer(model, test_loader, device, id2label)
    
    with open("transformer_results.txt", "w") as f:
        f.write("DistilBERT-base-cased Results\n")
        f.write("============================\n")
        f.write(test_report)
    
    logger.info(f"Final Transformer F1: {test_f1:.4f}. Results saved to transformer_results.txt")

if __name__ == "__main__":
    train()
