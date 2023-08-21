from ..preprocess import process_transformers
from transformers import PreTrainedTokenizer
from datasets import load_dataset, DatasetDict
from typing import Any

MAX_LENGTH = 416

def get_raw_downstream_dataset(name: str) -> DatasetDict:
    if name in ("wisesight_sentiment", "generated_reviews_enth", "prachathai67k"):
        return load_dataset(name, cache_dir="cache")
    elif name in ("lst20_pos", "lst20_ner"):
        return load_dataset("lst20", data_dir="LST20_Corpus", cache_dir="cache")
    elif name == "wongnai_reviews":
        unsplit = load_dataset("wongnai_reviews", cache_dir="cache")
        train_and_validation = unsplit["train"].train_test_split(train_size=36000, seed=2020)
        return DatasetDict(
            train=train_and_validation["train"],
            validation=train_and_validation["test"],
            test=unsplit["test"]
        )
    elif name == "thainer":
        unsplit = load_dataset("thainer", cache_dir="cache")
        labels: list = unsplit["train"].features["ner_tags"].feature.names
        old_id2label = dict(enumerate(labels))
        labels.remove("B-ไม่ยืนยัน")
        labels.remove("I-ไม่ยืนยัน")
        label2new_id = {label: id for id, label in enumerate(labels)}
        unsplit = unsplit.map(
            lambda examples: {
                "ner_tags": [
                    [
                        label2new_id.get(old_id2label[id], label2new_id["O"])
                        for id in ids
                    ] for ids in examples["ner_tags"]
                ]
            },
            batched=True
        )
        train_and_the_rest = unsplit["train"].train_test_split(train_size=5079, seed=2020)
        validation_and_test = train_and_the_rest["test"].train_test_split(train_size=635, seed=2020)
        return DatasetDict(
            train=train_and_the_rest["train"],
            validation=validation_and_test["train"],
            test=validation_and_test["test"]
        )
    else:
        raise ValueError(f"Invalid downstream dataset name: {name}")

def get_downstream_dataset(name: str, tokenizer: PreTrainedTokenizer):
    dataset = get_raw_downstream_dataset(name)
    if name == "wisesight_sentiment":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(text) for text in examples["texts"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns="texts"
        )
        return dataset.rename_column("category", "labels")
    elif name == "generated_reviews_enth":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(pair["th"]) for pair in examples["translation"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns=["translation", "correct"]
        )
        return dataset.rename_column("review_star", "labels")
    elif name == "wongnai_reviews":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(text) for text in examples["review_body"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns="review_body"
        )
        return dataset.rename_column("star_rating", "labels")
    elif name == "prachathai67k":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(text) for text in examples["title"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns=["url", "date", "title", "body_text"]
        )
        return dataset.map(
            lambda examples: {
                "labels": [
                    list(map(float, label))
                    for label in zip(
                        examples["politics"],
                        examples["human_rights"],
                        examples["quality_of_life"],
                        examples["international"],
                        examples["social"],
                        examples["environment"],
                        examples["economics"],
                        examples["culture"],
                        examples["labor"],
                        examples["national_security"],
                        examples["ict"],
                        examples["education"]
                    )
                ]
            },
            batched=True,
            remove_columns=[
                "politics",
                "human_rights",
                "quality_of_life",
                "international",
                "social",
                "environment",
                "economics",
                "culture",
                "labor",
                "national_security",
                "ict",
                "education"
            ]
        )
    elif name == "thainer":
        return dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer),
            batched=True,
            remove_columns=["id", "tokens"]
        )
    elif name in ("lst20_pos", "lst20_ner"):
        return dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer),
            batched=True,
            remove_columns=["clause_tags", "fname", "id", "tokens"]
        )
    else:
        raise ValueError(f"Invalid downstream dataset name: {name}")

def tokenize_and_align_labels(examples: dict[str, Any], tokenizer: PreTrainedTokenizer):
    tokenized_inputs = tokenizer(
        [
            [process_transformers(token) for token in tokens]
            for tokens in examples["tokens"]
        ],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    for tag in ("ner_tags", "pos_tags"):
        labels = []
        for i, label in enumerate(examples[tag]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs[tag] = labels
    return tokenized_inputs
