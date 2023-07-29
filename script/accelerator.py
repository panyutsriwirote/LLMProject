from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    is_torch_version,
    convert_outputs_to_fp32,
    has_transformer_engine_layers,
    convert_model,
    is_fp8_available,
    DynamoBackend,
    is_tpu_available
)
from types import MethodType
import torch, inspect

if is_fp8_available():
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast

if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

logger = get_logger(__name__)

class RepreparableAccelerator(Accelerator):

    def prepare_model(
        self,
        model: torch.nn.Module,
        device_placement: bool = None,
        evaluation_mode: bool = False
    ):
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP
        old_model_type = type(model)
        try:
            model_index = self._models.index(model)
            reprepare = True
        except ValueError:
            self._models.append(model)
            model_index = -1
            reprepare = False

        # We check only for models loaded with `accelerate`
        # Checks if any of the child module has the attribute `hf_device_map`.
        has_hf_device_map = False
        for m in model.modules():
            if hasattr(m, "hf_device_map"):
                has_hf_device_map = True
                break

        if (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)) and getattr(
            model, "hf_device_map", False
        ):
            model_devices = set(model.hf_device_map.values())
            if len(model_devices) > 1 and self.distributed_type != DistributedType.NO:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision on multiple devices in any distributed mode."
                    " In order to use 8-bit models that have been loaded across multiple GPUs the solution is to use Naive Pipeline Parallelism."
                    " Therefore you should not specify that you are under any distributed regime in your accelerate config."
                )
            current_device = list(model_devices)[0]
            current_device_index = current_device.index if isinstance(current_device, torch.device) else current_device

            if torch.device(current_device_index) != self.device:
                # if on the first device (GPU 0) we don't care
                if (self.device.index is not None) or (current_device_index != 0):
                    raise ValueError(
                        "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}"
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"
                    )

            if "cpu" in model_devices or "disk" in model_devices:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
                )
        elif device_placement and not has_hf_device_map:
            model = model.to(self.device)

        if not evaluation_mode:
            if self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_XPU):
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[self.local_process_index], output_device=self.local_process_index, **kwargs
                    )
            elif self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

                # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
                # don't wrap it again
                if type(model) != FSDP:
                    self.state.fsdp_plugin.set_auto_wrap_policy(model)
                    fsdp_plugin = self.state.fsdp_plugin
                    kwargs = {
                        "sharding_strategy": fsdp_plugin.sharding_strategy,
                        "cpu_offload": fsdp_plugin.cpu_offload,
                        "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                        "backward_prefetch": fsdp_plugin.backward_prefetch,
                        "mixed_precision": fsdp_plugin.mixed_precision_policy,
                        "ignored_modules": fsdp_plugin.ignored_modules,
                        "device_id": self.device,
                    }
                    signature = inspect.signature(FSDP.__init__).parameters.keys()
                    if "limit_all_gathers" in signature:
                        kwargs["limit_all_gathers"] = fsdp_plugin.limit_all_gathers
                    if "use_orig_params" in signature:
                        kwargs["use_orig_params"] = fsdp_plugin.use_orig_params
                    model = FSDP(model, **kwargs)
                self._models[model_index] = model
            elif self.distributed_type == DistributedType.MULTI_CPU:
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        if self.native_amp:
            model._original_forward = model.forward
            if self.mixed_precision == "fp16" and is_torch_version(">=", "1.10"):
                model.forward = MethodType(torch.cuda.amp.autocast(dtype=torch.float16)(model.forward.__func__), model)
            elif self.mixed_precision == "bf16" and self.distributed_type != DistributedType.TPU:
                model.forward = MethodType(
                    torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)(model.forward.__func__), model
                )
            else:
                model.forward = MethodType(torch.cuda.amp.autocast()(model.forward.__func__), model)
            model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
        elif self.mixed_precision == "fp8":
            if not has_transformer_engine_layers(model):
                with torch.no_grad():
                    convert_model(model)
                model._converted_to_transformer_engine = True
            model._original_forward = model.forward

            kwargs = self.fp8_recipe_handler.to_kwargs() if self.fp8_recipe_handler is not None else {}
            if "fp8_format" in kwargs:
                kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
            fp8_recipe = te_recipe.DelayedScaling(**kwargs)
            cuda_device_capacity = torch.cuda.get_device_capability()
            fp8_enabled = cuda_device_capacity[0] >= 9 or (
                cuda_device_capacity[0] == 8 and cuda_device_capacity[1] >= 9
            )
            if not fp8_enabled:
                logger.warn(
                    f"The current device has compute capability of {cuda_device_capacity} which is "
                    "insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace "
                    "or higher, compute capability of 8.9 or higher). Will use FP16 instead."
                )
            model.forward = fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe)(model.forward)
        if not evaluation_mode:
            if self.distributed_type == DistributedType.TPU and self.state.fork_launched:
                model = xmp.MpModelWrapper(model).to(self.device)
        # torch.compile should be called last.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO:
            if not is_torch_version(">=", "2.0"):
                raise ValueError("Using `torch.compile` requires PyTorch 2.0 or higher.")
            model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        new_model_type = type(model)
        if new_model_type is not old_model_type and self.is_main_process:
            model_type_name = new_model_type.__name__
            if reprepare:
                print(f"Reprepared '{model_type_name}'")
            else:
                print(f"Wrapped model into '{model_type_name}'")
        return model
