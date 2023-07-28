from script import (
    NewWangchanForMaskedLM,
    Config,
    get_layer_params,
    check_layer_is_exhaustive,
    get_optimizer_param_groups
)
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.distributed.elastic.multiprocessing.errors import record
from accelerate import Accelerator, find_executable_batch_size
from accelerate.utils import RNG_STATE_NAME
from tqdm.auto import tqdm
from argparse import ArgumentParser
from typing import Literal
from shutil import copyfile, SameFileError
from os import makedirs, listdir, path
from sys import stdout
import torch, re, random, numpy

CHECKPOINT_FORMAT = re.compile(r"step_\d+")

# Training function
def main(config: Config):
    # Extract config
    training_config = config.training_config
    optimizer_config = config.optimizer_config
    scheduler_config = config.scheduler_config
    unfreezing_config = config.unfreezing_config
    layer_config = config.layer_config
    script_config = config.script_config

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=training_config.mixed_precision,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps
    )

    # Prepare for training
    if accelerator.is_main_process:
        print("Start training with the following configuration:")
        print(config)
        if training_config.save_steps != "no" or script_config.continue_from_checkpoint:
            makedirs(script_config.model_dir, exist_ok=True)
            try:
                copyfile(script_config.config_file, path.join(script_config.model_dir, "last_config.toml"))
            except SameFileError:
                pass
    def print_on_main(string: str):
        if accelerator.is_main_process:
            print(string)

    @record
    @find_executable_batch_size(starting_batch_size=training_config.batch_size)
    def inner_train(batch_size: int):
        nonlocal accelerator
        accelerator.free_memory()
        print_on_main(
            "Try training with effective batch size = "
            f"per_device ({batch_size}) * "
            f"num_devices ({accelerator.num_processes}) * "
            f"gradient_accumulation_steps ({training_config.gradient_accumulation_steps})"
            f" = {batch_size * accelerator.num_processes * training_config.gradient_accumulation_steps}"
        )

        # Load model
        model = NewWangchanForMaskedLM.from_pretrained("model")

        # Validate layer config
        if layer_config is not None:
            check_layer_is_exhaustive(model, config)

        # Data loaders
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=AutoTokenizer.from_pretrained("model"),
            mlm_probability=0.15
        )
        def get_dataset(dataset_path: str) -> Dataset:
            if (
                path.isdir(dataset_path) and
                any(path.splitext(p)[1] == ".arrow" for p in listdir(dataset_path))
            ):
                dataset = load_from_disk(dataset_path)
            elif path.splitext(dataset_path)[1] == ".arrow":
                dataset = Dataset.from_file(dataset_path)
            else:
                raise ValueError(f"'{dataset_path}' is not a valid dataset path")
            print_on_main(f"Loaded {len(dataset)} examples from '{dataset_path}'")
            return dataset
        train_dataloader = DataLoader(
            dataset=get_dataset(script_config.train_data),
            shuffle=True,
            batch_size=batch_size,
            collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            dataset=get_dataset(script_config.eval_data),
            batch_size=batch_size,
            collate_fn=data_collator
        )

        # Optimizer
        optimizer = AdamW(
            params=get_optimizer_param_groups(model, config),
            lr=optimizer_config.peak_lr,
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.eps,
            betas=tuple(optimizer_config.betas)
        )

        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=scheduler_config.num_warmup_steps,
            num_training_steps=scheduler_config.max_steps
        )

        # Prepare for distributed training
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
        )
        print_on_main(f"Wrapped model into '{type(model).__name__}'")

        # Load from checkpoint
        if script_config.continue_from_checkpoint:
            latest_checkpoint = max(
                (
                p
                for p in listdir(script_config.model_dir)
                    if path.isdir(path.join(script_config.model_dir, p)) and CHECKPOINT_FORMAT.fullmatch(p)
                ),
                key = lambda p: int(p.split('_')[1])
            )
            checkpoint_dir = path.join(script_config.model_dir, latest_checkpoint)
            accelerator.load_state(checkpoint_dir)
            print_on_main(f"Loaded from checkpoint '{checkpoint_dir}'")
            step = int(latest_checkpoint.split('_')[1])
        else:
            step = 0

        # Calculate steps
        num_epochs = training_config.num_epochs
        num_batches_per_epoch = len(train_dataloader)
        num_update_steps_per_epoch = -(num_batches_per_epoch // -training_config.gradient_accumulation_steps)
        num_training_steps = num_epochs * num_update_steps_per_epoch
        epoch, num_skipped = divmod(step, num_update_steps_per_epoch)
        if num_skipped != 0:
            epoch += 1

        def get_action_step(step_config: Literal["no", "once", "epoch"] | int):
            if step_config == "no":
                return 0
            elif step_config == "epoch":
                return num_update_steps_per_epoch
            elif step_config == "once":
                return num_training_steps
            else:
                return step_config
        eval_steps = get_action_step(training_config.eval_steps)
        save_steps = get_action_step(training_config.save_steps)

        # Gradual unfreezing
        def get_unfreeze_func():
            if unfreezing_config is None:
                return lambda _: None
            else:
                # Freeze everything
                for param in model.parameters():
                    param.requires_grad = False
                print_on_main("Freezed all parameters")
                # Define unfreezing function
                layers = layer_config.layers
                schedule = unfreezing_config.schedule
                if unfreezing_config.mode == "epoch":
                    schedule = (n * num_update_steps_per_epoch for n in schedule)
                step_to_layer = dict(zip(schedule, layers))
                unwrapped_model = accelerator.unwrap_model(model)
                def unfreeze(step: int):
                    if step in step_to_layer:
                        layer = step_to_layer[step]
                        for param in get_layer_params(unwrapped_model, layer):
                            param.requires_grad = True
                        del step_to_layer[step]
                        print_on_main(f"Unfreezed {layer}")
                # Unfreeze already-unfreezed layers
                for i in sorted(step_to_layer):
                    if i < step:
                        unfreeze(i)
                    else:
                        break
                return unfreeze
        unfreeze = get_unfreeze_func()

        # Evaluation
        def eval():
            model.eval()
            losses = []
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(batch_size)))
            losses = torch.cat(losses)
            print_on_main(f">>> Step {step} (Epoch {epoch}): Mean Loss: {torch.mean(losses)}")
            model.train()

        # Training loop
        progress_bar = tqdm(initial=step, total=num_training_steps, file=stdout, disable=not accelerator.is_main_process)
        def train_epoch(dataloader: DataLoader):
            nonlocal step
            model.train()
            for batch in dataloader:
                # Unfreeze
                unfreeze(step)
                with accelerator.accumulate(model):
                    # Forward
                    outputs = model(**batch)
                    # Backward
                    accelerator.backward(outputs.loss)
                    # Step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        # Progress
                        progress_bar.update(1)
                        step += 1
                        # Evaluation
                        if eval_steps and step % eval_steps == 0:
                            eval()
                        # Save model
                        if save_steps and step % save_steps == 0:
                            accelerator.wait_for_everyone()
                            checkpoint_dir = path.join(script_config.model_dir, f"step_{step}")
                            if accelerator.is_main_process:
                                accelerator.save_state(checkpoint_dir)
                                print(f"Saved checkpoint to '{checkpoint_dir}'")
                                # Random state of main process is already saved here
                            else:
                                # Save random state of other processes
                                makedirs(checkpoint_dir, exist_ok=True)
                                states = {}
                                states_name = f"{RNG_STATE_NAME}_{accelerator.state.process_index}.pkl"
                                states["random_state"] = random.getstate()
                                states["numpy_random_seed"] = numpy.random.get_state()
                                states["torch_manual_seed"] = torch.get_rng_state()
                                states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
                                output_states_file = path.join(checkpoint_dir, states_name)
                                torch.save(states, output_states_file)

        # Evaluation before training
        eval()

        # First epoch
        if num_skipped != 0:
            print_on_main(f"Skipped first {num_skipped} steps in train dataloader")
            skipped_dataloader = accelerator.skip_first_batches(
                train_dataloader,
                num_skipped * training_config.gradient_accumulation_steps
            )
            train_epoch(skipped_dataloader)
        epoch += 1

        # Reamining epochs
        for epoch in range(epoch, num_epochs+1):
            train_epoch(train_dataloader)

        print_on_main("Training finished according to configuration")

    try:
        inner_train()
    except Exception as e:
        print_on_main(e)
        raise

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--continue_from_checkpoint", action="store_true")
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    config = Config.from_args(args)
    main(config)
