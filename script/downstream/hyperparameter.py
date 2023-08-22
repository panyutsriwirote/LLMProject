from .datasets import get_downstream_dataset
from .finetuning import DATASET_NAME_TO_TASK, f1_metrics
from optuna import Trial
from os import path
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)

def hp_space(trial: Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 10),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64])
    }

def hp_search_on_dataset(name: str, model_dir: str):
    # Get dataset
    tokenizer = AutoTokenizer.from_pretrained("model")
    dataset, id2label = get_downstream_dataset(name, tokenizer)
    # Get model
    task = DATASET_NAME_TO_TASK[name]
    if task in ("named_entity_recognition", "token_classification"):
        model_init = lambda _: AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=len(id2label)
        )
    else:
        model_init = lambda _: AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            problem_type=task,
            num_labels=len(id2label)
        )
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
    training_args = TrainingArguments(
        output_dir=path.join("hp_search", name),
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
    # Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=f1_metrics(task, id2label)
    )
    return trainer.hyperparameter_search(
        hp_space=hp_space,
        backend="optuna",
        n_trials=20,
        compute_objective=lambda metrics: metrics[metric_for_best_model],
        direction="minimize" if metric_for_best_model == "eval_loss" else "maximize"
    )
