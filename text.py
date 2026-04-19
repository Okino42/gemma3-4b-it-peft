import torch
from torch.optim import AdamW
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json

# import data
csv_path = "/projects/_hdd/checkpoint/all-data.csv"
df = pd.read_csv(
    csv_path,
    header=None,
    names=["label", "sentence"], # add column names for dataset
    encoding="latin1",
    lineterminator="\r"
)

# data preprocessing
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["sentence"] = (
    df["sentence"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
) # remove unnecessary spaces
df = df[(df["sentence"] != "") & (df["label"] != "")].copy() # remove empty samples
valid_labels = {"negative", "neutral", "positive"}
df = df[df["label"].isin(valid_labels)].copy() # only retain three kinds of labels
df = df.drop_duplicates(subset=["label", "sentence"]).reset_index(drop=True) # remove duplicated samples
conflict_sentences = df.groupby("sentence")["label"].nunique() # remove label conflict sample
conflict_sentences = conflict_sentences[conflict_sentences > 1].index
df = df[~df["sentence"].isin(conflict_sentences)].reset_index(drop=True)
print("Cleaned dataset size:", len(df))
print(df["label"].value_counts())

# label mapping
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["label"].map(label2id)

# splitting by using stratified sampling
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)
print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# model and tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# tokenizing and truncation
def preprocess(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=128,
    )
train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

# Remove columns that are not consumed by Trainer.
remove_cols = [col for col in train_ds.column_names if col not in ["input_ids", "attention_mask", "label"]]
train_ds = train_ds.remove_columns(remove_cols)
val_ds = val_ds.remove_columns(remove_cols)
test_ds = test_ds.remove_columns(remove_cols)

train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # use dynamic padding

# evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }
def save_eval_artifacts(trainer, dataset, split_name, output_dir, id2label):
    os.makedirs(output_dir, exist_ok=True)

    predictions = trainer.predict(dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)

    labels = sorted(id2label.keys())
    target_names = [id2label[i] for i in labels]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=6,
        zero_division=0,
        output_dict=True,
    )

    # save results
    with open(os.path.join(output_dir, f"{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # save confusion matrix
    cm_payload = {
        "labels": target_names,
        "matrix": cm.tolist(),
    }
    with open(os.path.join(output_dir, f"{split_name}_confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, ensure_ascii=False, indent=2)

    # save classification report
    with open(os.path.join(output_dir, f"{split_name}_classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # save results of every samples
    rows = []
    for i in range(len(y_true)):
        rows.append({
            "gold_label_id": int(y_true[i]),
            "gold_label": id2label[int(y_true[i])],
            "pred_label_id": int(y_pred[i]),
            "pred_label": id2label[int(y_pred[i])],
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(
        os.path.join(output_dir, f"{split_name}_predictions.csv"),
        index=False,
        encoding="utf-8",
    )

    print(f"\n[{split_name}] metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n[{split_name}] confusion matrix:")
    print(cm)

# layer-wise lr decay
def get_llrd_optimizer(model, base_lr=2e-5, layer_decay=0.9, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    optimizer_grouped_parameters = []

    # set highest lr for classification head
    head_lr = base_lr
    head_decay_params = []
    head_no_decay_params = []

    for n, p in model.classifier.named_parameters():
        if not p.requires_grad:
            continue
        full_name = f"classifier.{n}"
        if any(nd in full_name for nd in no_decay):
            head_no_decay_params.append(p)
        else:
            head_decay_params.append(p)

    optimizer_grouped_parameters.append(
        {"params": head_decay_params, "lr": head_lr, "weight_decay": weight_decay}
    )
    optimizer_grouped_parameters.append(
        {"params": head_no_decay_params, "lr": head_lr, "weight_decay": 0.0}
    )

    # lr for encoder layer
    n_layers = model.config.num_hidden_layers  # roberta-base = 12

    for layer_idx in range(n_layers - 1, -1, -1):
        layer_lr = base_lr * (layer_decay ** (n_layers - 1 - layer_idx))

        decay_params = []
        no_decay_params = []

        for n, p in model.roberta.encoder.layer[layer_idx].named_parameters():
            if not p.requires_grad:
                continue
            full_name = f"roberta.encoder.layer.{layer_idx}.{n}"
            if any(nd in full_name for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer_grouped_parameters.append(
            {"params": decay_params, "lr": layer_lr, "weight_decay": weight_decay}
        )
        optimizer_grouped_parameters.append(
            {"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0}
        )

    # set the lowest lr for embedding layer
    embeddings_lr = base_lr * (layer_decay ** n_layers)

    emb_decay_params = []
    emb_no_decay_params = []

    for n, p in model.roberta.embeddings.named_parameters():
        if not p.requires_grad:
            continue
        full_name = f"roberta.embeddings.{n}"
        if any(nd in full_name for nd in no_decay):
            emb_no_decay_params.append(p)
        else:
            emb_decay_params.append(p)

    optimizer_grouped_parameters.append(
        {"params": emb_decay_params, "lr": embeddings_lr, "weight_decay": weight_decay}
    )
    optimizer_grouped_parameters.append(
        {"params": emb_no_decay_params, "lr": embeddings_lr, "weight_decay": 0.0}
    )

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    return optimizer

# training hyperparameter
# ==================== 8. Training arguments ====================
args = TrainingArguments(
    output_dir="/projects/_hdd/checkpoint/roberta_fpb_llrd",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=10,
    bf16=True,
    seed=42,
    dataloader_num_workers=4,
    save_total_limit=2,
    report_to="none",
)

# ==================== 9. LLRD switch ====================
use_llrd = True
base_lr = 1e-4
layer_decay = 0.9
warmup_ratio = 0.1

steps_per_epoch = len(train_ds) // args.per_device_train_batch_size
steps_per_epoch = max(steps_per_epoch, 1)
num_training_steps = steps_per_epoch * args.num_train_epochs
num_warmup_steps = int(warmup_ratio * num_training_steps)

optimizer = None
scheduler = None

if use_llrd:
    print(f"Using LLRD: base_lr={base_lr}, layer_decay={layer_decay}")
    optimizer = get_llrd_optimizer(
        model,
        base_lr=base_lr,
        layer_decay=layer_decay,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
else:
    print(f"Using default Trainer optimizer: learning_rate={args.learning_rate}")

# ==================== 10. Trainer ====================
trainer_kwargs = dict(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if use_llrd:
    trainer_kwargs["optimizers"] = (optimizer, scheduler)

trainer = Trainer(**trainer_kwargs)

# 9. train and test
trainer.train()

print("Validation metrics:")
print(trainer.evaluate(eval_dataset=val_ds))

print("Test metrics:")
print(trainer.evaluate(eval_dataset=test_ds))

analysis_dir = os.path.join(args.output_dir, "analysis")
os.makedirs(analysis_dir, exist_ok=True)

save_eval_artifacts(
    trainer=trainer,
    dataset=val_ds,
    split_name="validation",
    output_dir=analysis_dir,
    id2label=id2label,
)

save_eval_artifacts(
    trainer=trainer,
    dataset=test_ds,
    split_name="test",
    output_dir=analysis_dir,
    id2label=id2label,
)
