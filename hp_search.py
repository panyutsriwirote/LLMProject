from script import get_downstream_dataset, hp_search_on_dataset
from transformers import AutoTokenizer

trained_model_step = 51500
tokenizer = AutoTokenizer.from_pretrained("model")

for dataset_name in (
    "wisesight_sentiment",
    "wongnai_reviews",
    "generated_reviews_enth",
    "prachathai67k",
    "thainer",
    "lst20_pos",
    "lst20_ner"
):
    dataset, id2label = get_downstream_dataset(dataset_name, tokenizer)
    best_run = hp_search_on_dataset(
        dataset=dataset,
        id2label=id2label,
        model_dir=f"trained_models/{trained_model_step}",
        tokenizer=tokenizer
    )
    print("*************************")
    print(f"Best run for {dataset}:")
    print(best_run)
    print("*************************")
