from .downstream_datasets import get_downstream_dataset
from typing import Literal
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    EvalPrediction
)
import evaluate, torch

def f1_metrics(task: Literal["single_label_classification", "multi_label_classification", "token_classification"]):
    f1 = evaluate.load("f1", "multilabel" if task == "multi_label_classification" else None)
    def compute_metrics(p: EvalPrediction):
        predictions, label_ids = p
        if task == "single_label_classification":
            predictions = predictions.argmax(axis=1)
        elif task == "multi_label_classification":
            predictions = torch.nn.Sigmoid()(torch.tensor(predictions)) > 0.5
        elif task == "token_classification":
            predictions = predictions.argmax(axis=2).flatten()
            label_ids = label_ids.flatten()
            predictions = [p for p, l in zip(predictions, label_ids) if l != -100]
            label_ids = [l for l in label_ids if l != -100]
        micro_f1 = f1.compute(predictions=predictions, references=label_ids, average="micro")["f1"]
        macro_f1 = f1.compute(predictions=predictions, references=label_ids, average="macro")["f1"]
        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }
    return compute_metrics

DATASET_NAME_TO_TAKS = {
    "wisesight_sentiment": "single_label_classification",
    "generated_reviews_enth": "single_label_classification",
    "wongnai_reviews": "single_label_classification",
    "prachathai67k": "multi_label_classification",
    "thainer": "token_classification",
    "lst20_pos": "token_classification",
    "lst20_ner": "token_classification"
}

def finetune_on_dataset(name: str, model_dir: str, training_args: TrainingArguments):
    # Get dataset
    tokenizer = AutoTokenizer.from_pretrained("model")
    dataset = get_downstream_dataset(name, tokenizer)
    # Get model
    task = DATASET_NAME_TO_TAKS[name]
    if name in ("wisesight_sentiment", "generated_reviews_enth", "wongnai_reviews"):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            problem_type=task,
            num_labels=dataset["train"].features["labels"].num_classes if name != "generated_reviews_enth" else 5
        )
    elif name == "prachathai67k":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            problem_type=task,
            num_labels=12
        )
    elif name == "thainer":
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=dataset["train"].features["ner_tags"].feature.num_classes - 2 # B-ไม่ยืนยัน and I-ไม่ยืนยัน is removed
        )
        dataset = dataset.remove_columns("pos_tags").rename_column("ner_tags", "labels")
    elif name == "lst20_pos":
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=dataset["train"].features["pos_tags"].feature.num_classes
        )
        dataset = dataset.remove_columns("ner_tags").rename_column("pos_tags", "labels")
    elif name == "lst20_ner":
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=dataset["train"].features["ner_tags"].feature.num_classes
        )
        dataset = dataset.remove_columns("pos_tags").rename_column("ner_tags", "labels")
    else:
        raise ValueError(f"Invalid downstream dataset name: {name}")
    # Data collator
    if task == "token_classification":
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=f1_metrics(task)
    )
    trainer.train()
    trainer.evaluate(eval_dataset=dataset["test"])
