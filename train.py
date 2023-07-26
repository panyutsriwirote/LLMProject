from script.new_wangchan import NewWangchanForMaskedLM
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler, PreTrainedModel
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size, RNG_STATE_NAME
from tqdm.auto import tqdm
from argparse import ArgumentParser, Namespace
from typing import Literal, TypedDict
from functools import reduce
from shutil import copyfile, SameFileError
from os import makedirs, listdir, path
from sys import stdout
from dataclasses import dataclass
import torch, tomllib, re, random, numpy

@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    eval_steps: Literal["no", "epoch", "once"] | int
    save_steps: Literal["no", "epoch", "once"] | int
    gradient_accumulation_steps: int
    mixed_precision: str

    def __str__(self):
        return f"{self.__class__.__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class OptimizerConfig:
    peak_lr: float
    weight_decay: float
    eps: float
    betas: tuple[float, float]
    layer_lr_decay_factor: int | None

    def __str__(self):
        return f"{self.__class__.__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class SchedulerConfig:
    num_warmup_steps: int
    max_steps: int

    def __str__(self):
        return f"{self.__class__.__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class UnfreezingConfig:
    mode: Literal["epoch", "step"]
    schedule: list[int]

    def __str__(self):
        return f"{self.__class__.__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class Layer:
    include: list[str]
    exclude: list[str] | None = None

    def __str__(self):
        return f"{self.include}{f' - {self.exclude}' if self.exclude else ''}"

@dataclass
class LayerConfig:
    layers: list[Layer]

    def __str__(self):
        return f"{self.__class__.__name__}(\n" + '\n'.join(f"    {layer}" for layer in self.layers) + "\n)"

@dataclass
class ScriptConfig:
    model_dir: str | None
    config_file: str
    train_data: str
    eval_data: str
    continue_from_checkpoint: bool

    def __str__(self):
        return f"{self.__class__.__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class Config:
    training_config: TrainingConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    unfreezing_config: UnfreezingConfig | None
    layer_config: LayerConfig | None
    script_config: ScriptConfig

    def __str__(self):
        return '\n'.join(str(v) for v in vars(self).values() if v is not None)

    @classmethod
    def from_args(cls, args: Namespace):
        if args.config_file is None:
            if args.model_dir is None:
                raise ValueError("Cannot determine config file path. Please specify --model_dir or --config_file")
            args.config_file = path.join(args.model_dir, "last_config.toml")
        script_config = ScriptConfig(**vars(args))
        with open(script_config.config_file, "rb") as f:
            config = tomllib.load(f)
        config = cls(
            training_config=TrainingConfig(**config["training"]),
            optimizer_config=OptimizerConfig(**config["optimizer"]),
            scheduler_config=SchedulerConfig(**config["scheduler"]),
            unfreezing_config=UnfreezingConfig(**config["unfreezing"]) if "unfreezing" in config else None,
            layer_config=LayerConfig(
                [Layer(**layer) for layer in config["layer"]]
            ) if "layer" in config else None,
            script_config=script_config
        )
        if (
            config.unfreezing_config is not None or
            config.optimizer_config.layer_lr_decay_factor is not None
        ) and config.layer_config is None:
            raise ValueError("Must specify 'layer' when using 'gradual unfreezing' or 'discriminative fine-tuning'")
        if (
            config.training_config.save_steps != "no" or
            script_config.continue_from_checkpoint
        ) and config.script_config.model_dir is None:
            raise ValueError("Cannot determine model directory. Please specify --model_dir")
        if (
            config.unfreezing_config is not None and
            len(config.layer_config.layers) != len(config.unfreezing_config.schedule)
        ):
            raise ValueError(
                f"Length of 'layer' and 'schedule' must be the same. "
                f"Got {len(config.layer_config.layers)} and {len(config.unfreezing_config.schedule)}"
            )
        if (
            config.training_config.eval_steps not in ("no", "epoch", "once") and
            not isinstance(config.training_config.eval_steps, int)
        ):
            raise ValueError(
                f"'eval_steps' must be one of 'no', 'epoch', 'once' or an integer. "
                f"Got {config.training_config.eval_steps}"
            )
        if (
            config.training_config.save_steps not in ("no", "epoch", "once") and
            not isinstance(config.training_config.save_steps, int)
        ):
            raise ValueError(
                f"'save_steps' must be one of 'no', 'epoch', 'once' or an integer. "
                f"Got {config.training_config.save_steps}"
            )
        return config

class ParameterGroup(TypedDict):
    params: list[Parameter]
    lr: float

CHECKPOINT_FORMAT = re.compile(r"step_\d+")

# Util functions
def get_module(model: PreTrainedModel, name: str) -> Module | Parameter:
    return reduce(getattr, name.split('.'), model)
def get_params(model: PreTrainedModel, name: str):
    module = get_module(model, name)
    if isinstance(module, Parameter):
        yield module
    else:
        for param in module.parameters():
            yield param
def get_layer_params(model: PreTrainedModel, layer: Layer):
    include = {param for name in layer.include for param in get_params(model, name)}
    exclude = {param for name in layer.exclude for param in get_params(model, name)} if layer.exclude is not None else set()
    return include - exclude

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
            def check_layer_is_exhaustive():
                accounted_for: set[Parameter] = set()
                for layer in layer_config.layers:
                    accounted_for.update(get_layer_params(model, layer))
                num_unaccounted_for = len(set(model.parameters()) - accounted_for)
                if num_unaccounted_for != 0:
                    raise ValueError(
                        f"'layer' defined in {script_config.config_file} is not exhaustive."
                        f" {num_unaccounted_for} parameters are not accounted for."
                    )
            check_layer_is_exhaustive()

        # Freeze everything
        if unfreezing_config is not None:
            for param in model.parameters():
                param.requires_grad = False
            print_on_main("Freezed all parameters")

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
        def get_optimizer_param_groups():
            if optimizer_config.layer_lr_decay_factor is None:
                return model.parameters()
            else:
                # Discriminative fine-tuning
                params: list[ParameterGroup] = []
                param_set: set[Parameter] = set()
                current_lr = optimizer_config.peak_lr
                decay_factor = optimizer_config.layer_lr_decay_factor
                for layer in layer_config.layers:
                    param_group = ParameterGroup(params=[], lr=current_lr)
                    for param in get_layer_params(model, layer):
                        if param not in param_set:
                            param_group["params"].append(param)
                            param_set.add(param)
                    params.append(param_group)
                    current_lr /= decay_factor
                return params

        optimizer = AdamW(
            params=get_optimizer_param_groups(),
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
        unwrapped_model = accelerator.unwrap_model(model)

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
        progress_bar = tqdm(initial=step, total=num_training_steps, file=stdout, disable=not accelerator.is_main_process)
        epoch, num_skipped = divmod(step, num_update_steps_per_epoch)

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
                return lambda: None
            else:
                layers = layer_config.layers
                schedule = unfreezing_config.schedule
                if unfreezing_config.mode == "epoch":
                    schedule = (n * num_update_steps_per_epoch for n in schedule)
                step_to_param = dict(zip(schedule, layers))
                def unfreeze():
                    if step in step_to_param:
                        layer = step_to_param[step]
                        for param in get_layer_params(unwrapped_model, layer):
                            param.requires_grad = True
                        del step_to_param[step]
                        print_on_main(f"Unfreezed {layer}")
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

        # Training loop
        def train_epoch(dataloader: DataLoader):
            nonlocal step
            model.train()
            for batch in dataloader:
                with accelerator.accumulate(model):
                    # Unfreeze
                    unfreeze()
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

        # First epoch if skipped
        if num_skipped:
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
