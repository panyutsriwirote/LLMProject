from .datasets import get_downstream_dataset
from typing import Literal
from os import path
from seqeval.metrics import classification_report as seqeval_metric
from sklearn.metrics import classification_report as sklearn_metric
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
import torch

Task = Literal[
    "single_label_classification",
    "multi_label_classification",
    "token_classification",
    "named_entity_recognition"
]

def f1_metrics(
    task: Task,
    id2label: dict[int, str]
):
    # Function that gets predictions and labels from EvalPrediction
    if task == "single_label_classification":
        preprocess = lambda eval_pred: (
            eval_pred.predictions.argmax(axis=1),
            eval_pred.label_ids
        )
    elif task == "multi_label_classification":
        preprocess = lambda eval_pred: (
            torch.nn.Sigmoid()(torch.tensor(eval_pred.predictions)) > 0.5,
            eval_pred.label_ids
        )
    elif task in ("token_classification", "named_entity_recognition"):
        def preprocess(eval_pred: EvalPrediction):
            predictions = eval_pred.predictions.argmax(axis=2).flatten()
            labels = eval_pred.label_ids.flatten()
            predictions = [id2label[p] for p, l in zip(predictions, labels) if l != -100]
            labels = [id2label[l] for l in labels if l != -100]
            return predictions, labels
    else:
        raise ValueError(f"Invalid task: {task}")
    # Function that computes metrics
    if task == "named_entity_recognition":
        metric = seqeval_metric
    else:
        metric = sklearn_metric
    # Function that converts result to Huggingface's format
    if task in ("single_label_classification", "token_classification"):
        postprocess = lambda result: {
            "micro_average_f1": result["accuracy"],
            "macro_average_f1": result["macro avg"]["f1-score"],
            "class_f1": {
                id2label[i]: result[i]["f1-score"]
                for i in result
                    if i.isdigit()
            }
        }
    elif task == "named_entity_recognition":
        tag_set = {tag[2:] for tag in id2label.values() if tag != "O"}
        postprocess = lambda result: {
            "micro_average_f1": result["micro avg"]["f1-score"],
            "macro_average_f1": result["macro avg"]["f1-score"],
            "class_f1": {
                tag: result[tag]["f1-score"]
                for tag in result
                    if tag in tag_set
            }
        }
    else:
        postprocess = lambda result: {
            "micro_average_f1": result["micro avg"]["f1-score"],
            "macro_average_f1": result["macro avg"]["f1-score"],
            "class_f1": {
                id2label[i]: result[i]["f1-score"]
                for i in result
                    if i.isdigit()
            }
        }
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels = preprocess(eval_pred)
        result = metric(y_pred=predictions, y_true=labels, output_dict=True)
        return postprocess(result)
    return compute_metrics

DATASET_NAME_TO_TASK: dict[str, Task] = {
    "wisesight_sentiment": "single_label_classification",
    "generated_reviews_enth": "single_label_classification",
    "wongnai_reviews": "single_label_classification",
    "prachathai67k": "multi_label_classification",
    "thainer": "named_entity_recognition",
    "lst20_pos": "token_classification",
    "lst20_ner": "named_entity_recognition"
}

def finetune_on_dataset(name: str, model_dir: str, override_default: dict[str] | None = None):
    # Get dataset
    tokenizer = AutoTokenizer.from_pretrained("model")
    dataset, id2label = get_downstream_dataset(name, tokenizer)
    # Get model
    task = DATASET_NAME_TO_TASK[name]
    if task in ("named_entity_recognition", "token_classification"):
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=len(id2label)
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            problem_type=task,
            num_labels=len(id2label)
        )
    # Set targets
    if task == "named_entity_recognition":
        dataset = dataset.remove_columns("pos_tags").rename_column("ner_tags", "labels")
    elif task == "token_classification":
        dataset = dataset.remove_columns("ner_tags").rename_column("pos_tags", "labels")
    # Data collator
    if task in ("named_entity_recognition", "token_classification"):
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Training arguments
    if task == "single_label_classification":
        metric_for_best_model = "eval_micro_average_f1"
    elif task == "multi_label_classification":
        metric_for_best_model = "eval_macro_average_f1"
    else:
        metric_for_best_model = "eval_loss"
    args = dict(
        output_dir=path.join("finetuned_models", name),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        per_device_train_batch_size=32 if task in ("named_entity_recognition", "token_classification") else 16,
        per_device_eval_batch_size=32 if task in ("named_entity_recognition", "token_classification") else 16,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        num_train_epochs=6 if task in ("named_entity_recognition", "token_classification") else 3,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model
    )
    if override_default is not None:
        args.update(override_default)
    training_args = TrainingArguments(**args)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=f1_metrics(task, id2label)
    )
    trainer.train()
    print("**Evaluate on test set**")
    print('\n'.join(f"{k}: {v}" for k, v in trainer.predict(test_dataset=dataset["test"]).metrics.items()))
    return trainer
