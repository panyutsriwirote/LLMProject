from script import hp_search_on_dataset

trained_model_step = 51500

for dataset in (
    "wisesight_sentiment",
    "wongnai_reviews",
    "generated_reviews_enth",
    "prachathai67k",
    "thainer",
    "lst20_pos",
    "lst20_ner"
):
    best_run = hp_search_on_dataset(
        dataset,
        f"trained_models/{trained_model_step}"
    )
    print("*************************")
    print(f"Best run for {dataset}:")
    print(best_run)
    print("*************************")
