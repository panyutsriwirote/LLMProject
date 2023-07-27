from dataclasses import dataclass
from argparse import Namespace
from os import path
from typing import Literal, TypedDict
from torch.nn import Module, Parameter
from transformers import PreTrainedModel
from functools import reduce
import tomllib, re

@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    eval_steps: Literal["no", "epoch", "once"] | int
    save_steps: Literal["no", "epoch", "once"] | int
    gradient_accumulation_steps: int
    mixed_precision: str

    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class OptimizerConfig:
    peak_lr: float
    weight_decay: float
    eps: float
    betas: tuple[float, float]
    layer_lr_decay_factor: int | None

    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class SchedulerConfig:
    num_warmup_steps: int
    max_steps: int

    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class UnfreezingConfig:
    mode: Literal["epoch", "step"]
    schedule: list[int]

    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

@dataclass
class Layer:
    include: list[str]
    exclude: list[str] | None = None

    def __str__(self):
        return f"{self.include}{f' - {self.exclude}' if self.exclude is not None else ''}"

@dataclass
class LayerConfig:
    layers: list[Layer]

    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {layer}" for layer in self.layers) + "\n)"

@dataclass
class ScriptConfig:
    model_dir: str | None
    config_file: str
    train_data: str
    eval_data: str
    continue_from_checkpoint: bool

    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {k}={repr(v)}" for k, v in vars(self).items()) + "\n)"

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

def check_layer_is_exhaustive(model: PreTrainedModel, config: Config):
    accounted_for: set[Parameter] = set()
    for layer in config.layer_config.layers:
        accounted_for.update(get_layer_params(model, layer))
    num_unaccounted_for = len(set(model.parameters()) - accounted_for)
    if num_unaccounted_for != 0:
        raise ValueError(
            f"'layer' defined in {config.script_config.config_file} is not exhaustive."
            f" {num_unaccounted_for} parameters are not accounted for."
        )

def get_optimizer_param_groups(model: PreTrainedModel, config: Config):
    optimizer_config = config.optimizer_config
    if optimizer_config.layer_lr_decay_factor is None:
        return model.parameters()
    else:
        # Discriminative fine-tuning
        params: list[ParameterGroup] = []
        param_set: set[Parameter] = set()
        current_lr = optimizer_config.peak_lr
        decay_factor = optimizer_config.layer_lr_decay_factor
        for layer in config.layer_config.layers:
            param_group = ParameterGroup(params=[], lr=current_lr)
            for param in get_layer_params(model, layer):
                if param not in param_set:
                    param_group["params"].append(param)
                    param_set.add(param)
            params.append(param_group)
            current_lr /= decay_factor
        return params
