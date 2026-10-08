import argparse
import hashlib
import importlib.util
import json
import math
import os
import gc
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from collections import Counter
import numpy as np
import torch

# Newer transformers versions may probe float8 dtypes that older torch builds do not expose.
# Provide safe fallbacks so model loading can proceed on older environments.
for _float8_name in ('float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz'):
    if not hasattr(torch, _float8_name):
        setattr(torch, _float8_name, torch.float16)

from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from merge_naming import canonicalize_merge_config, canonicalize_merge_space
from models.huggingface_clip import get_model_from_config as get_hf_clip_model_from_config


def prefer_local_hf_files():
    return any(
        str(value).lower() in {'1', 'true', 'yes'}
        for value in (
            os.getenv('HF_HUB_OFFLINE'),
            os.getenv('TRANSFORMERS_OFFLINE'),
            os.getenv('CORE_SPACE_MERGING_LOCAL_ONLY'),
        )
    )


def is_hf_connection_error(exc):
    message = str(exc)
    markers = (
        'HTTPSConnectionPool(',
        'MaxRetryError(',
        'SSLError',
        'ConnectionError',
        'ProxyError',
        'ReadTimeout',
        'ConnectTimeout',
        'Temporary failure in name resolution',
        'NameResolutionError',
        'TLS/SSL connection has been closed',
    )
    return any(marker in message for marker in markers)


def hf_load_with_local_fallback(loader, resource_name, **kwargs):
    if prefer_local_hf_files():
        kwargs = dict(kwargs)
        kwargs['local_files_only'] = True
        return loader(resource_name, **kwargs)
    try:
        return loader(resource_name, **kwargs)
    except Exception as exc:
        if not is_hf_connection_error(exc):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs['local_files_only'] = True
        print(f"Falling back to local Hugging Face cache for {resource_name}: {exc}")
        return loader(resource_name, **retry_kwargs)


def build_model_cache_signature(model_name, bases, ft_config=None):
    base_descriptors = []
    for base in bases:
        path = Path(base)
        descriptor = {'path': str(path)}
        if path.exists():
            stat = path.stat()
            descriptor.update({'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        base_descriptors.append(descriptor)
    payload = {
        'model_name': model_name,
        'bases': base_descriptors,
        'ft_config': ft_config or {},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def load_lora_state_dict(
    base_path,
    base_model=None,
    *,
    cache_dir=None,
    local_files_only=False,
):
    if base_path.endswith('.pt'):
        return torch.load(base_path, map_location='cpu')

    try:
        from peft.utils import load_peft_weights

        adapter_state_dict = hf_load_with_local_fallback(
            load_peft_weights,
            base_path,
            device='cpu',
            **_hf_download_kwargs(
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ),
        )
        if adapter_state_dict:
            return adapter_state_dict
    except Exception as exc:
        print(f"Falling back to PEFT adapter attach for {base_path}: {exc}")

    if base_model is None:
        raise RuntimeError(f"Unable to load LoRA weights for {base_path} without a base model fallback.")

    adapter_model = hf_load_with_local_fallback(
        lambda model_id, **kwargs: PeftModel.from_pretrained(model=base_model, model_id=model_id, **kwargs),
        base_path,
        **_hf_download_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ),
    )
    try:
        return get_peft_model_state_dict(adapter_model)
    finally:
        del adapter_model
        gc.collect()


def _hf_download_kwargs(*, cache_dir=None, local_files_only=False):
    """Return the Hub arguments shared by PEFT config and weight downloads."""
    kwargs = {}
    if cache_dir:
        kwargs['cache_dir'] = str(cache_dir)
    if local_files_only:
        kwargs['local_files_only'] = True
    return kwargs


def _load_peft_adapter_artifacts(
    adapter_source,
    *,
    cache_dir=None,
    local_files_only=False,
):
    """Load a PEFT config and its small adapter state without injecting modules."""
    from peft import PeftConfig
    from peft.utils import load_peft_weights

    download_kwargs = _hf_download_kwargs(
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    adapter_config = hf_load_with_local_fallback(
        lambda model_id, **kwargs: PeftConfig.from_pretrained(
            model_id,
            **kwargs,
        ),
        str(adapter_source),
        **download_kwargs,
    )
    state_dict = hf_load_with_local_fallback(
        load_peft_weights,
        str(adapter_source),
        device='cpu',
        **download_kwargs,
    )
    if not state_dict:
        raise ValueError(
            f"PEFT adapter contains no weights: {adapter_source}"
        )
    return adapter_config, state_dict


def _normalize_vision_lora_state_dict(state_dict):
    """Map repository exports and native PEFT adapters to one key namespace."""
    normalized = {}
    for raw_key, tensor in state_dict.items():
        key = str(raw_key).replace('.base_layer', '')
        if key.startswith('lora_model.'):
            key = f"vision_model.{key[len('lora_model.'):]}"
        elif key.startswith('base_model.'):
            # Native PEFT vision adapters are saved relative to the wrapped
            # CLIPVisionTransformer. The merge carrier lives one level higher.
            key = f'vision_model.{key}'
        elif key.startswith('linear_layer.'):
            key = f"vision_head.{key[len('linear_layer.'):]}"
        if key in normalized:
            raise ValueError(
                f"Vision adapter keys collide after normalization: {raw_key!r} "
                f"and another key both map to {key!r}"
            )
        normalized[key] = tensor
    return normalized


def _validate_peft_vision_adapter_config(
    adapter_config,
    *,
    adapter_source,
    expected_rank,
    expected_alpha,
    expected_target_modules,
    expected_base_model=None,
):
    """Reject Hub adapters whose PEFT metadata changes merge semantics."""
    errors = []
    peft_type = getattr(adapter_config, 'peft_type', None)
    peft_type = getattr(peft_type, 'value', peft_type)
    if str(peft_type).upper() != 'LORA':
        errors.append(f"expected peft_type LORA, got {peft_type!r}")

    actual_rank = getattr(adapter_config, 'r', None)
    if (
        expected_rank is not None
        and actual_rank is not None
        and int(actual_rank) != int(expected_rank)
    ):
        errors.append(
            f"expected rank {expected_rank}, got {actual_rank}"
        )

    actual_alpha = getattr(adapter_config, 'lora_alpha', None)
    if (
        expected_alpha is not None
        and actual_alpha is not None
        and float(actual_alpha) != float(expected_alpha)
    ):
        errors.append(
            f"expected lora_alpha {expected_alpha}, got {actual_alpha}"
        )
    if actual_rank is not None and actual_alpha is not None:
        actual_rank_float = float(actual_rank)
        if actual_rank_float <= 0:
            errors.append(f"expected a positive rank, got {actual_rank}")
        elif not math.isclose(
            float(actual_alpha) / actual_rank_float,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(
                "this merge path consumes raw B@A and requires "
                f"lora_alpha/r=1, got {actual_alpha}/{actual_rank}"
            )

    expected_targets = set(expected_target_modules or ())
    actual_targets = set(getattr(adapter_config, 'target_modules', ()) or ())
    if expected_targets and actual_targets != expected_targets:
        errors.append(
            f"expected target_modules {sorted(expected_targets)}, "
            f"got {sorted(actual_targets)}"
        )

    configured_base = getattr(
        adapter_config,
        'base_model_name_or_path',
        None,
    )
    if (
        configured_base
        and expected_base_model
        and not Path(str(expected_base_model)).is_absolute()
        and str(configured_base) != str(expected_base_model)
    ):
        errors.append(
            f"expected base model {expected_base_model!r}, "
            f"got {configured_base!r}"
        )

    if errors:
        raise ValueError(
            f"Invalid PEFT vision adapter config {adapter_source}:\n"
            + "\n".join(errors)
        )



##########################################################################################################################
######################################################### CLASSES ########################################################
##########################################################################################################################

def evaluate_logits(model, loader, device, mask_class=None, eval=True):
    """Evaluate a model trained with standard CE on a dataset."""
    hf_device_map = getattr(model, 'hf_device_map', None)
    if hf_device_map:
        input_device = next(
            (
                (
                    torch.device(f'cuda:{module_device}') if isinstance(module_device, int)
                    else module_device if isinstance(module_device, torch.device)
                    else torch.device(module_device)
                )
                for module_device in hf_device_map.values()
                if (
                    isinstance(module_device, int)
                    or isinstance(module_device, torch.device)
                    or (isinstance(module_device, str) and module_device.startswith('cuda'))
                )
            ),
            torch.device('cpu'),
        )
    else:
        model.to(device)
        input_device = torch.device(device)
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(loader, 'Evaluating model'):
        batch = batch.to(input_device)
        labels = batch["labels"]
        model_inputs = {key: value for key, value in batch.items() if key != "labels"}
        with torch.no_grad():
            outputs = model(**model_inputs)
            if mask_class is not None:
                if eval:
                    outputs.logits[:, mask_class] = -np.inf
                else:
                    outputs.logits[:, mask_class] = -1e10
            predictions = outputs.logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (predictions.cpu() == labels.cpu()).sum().item()
    return correct / total


def resolve_torch_dtype(dtype_name):
    if dtype_name is None:
        return torch.float16
    normalized = dtype_name.lower()
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[normalized]


def _print_device_map_summary(device_map):
    device_counts = Counter(str(device) for device in device_map.values())
    summary = ", ".join(
        f"{device}={count}" for device, count in sorted(device_counts.items(), key=lambda item: item[0])
    )
    print(f"Model parallel device map summary: {summary}")
    if any(device in {'cpu', 'disk'} for device in device_counts):
        print("Warning: some model weights were offloaded outside GPU; this usually makes eval much slower.")


def prepare_model_for_parallel_eval(
        model,
        use_model_parallel=False,
        dtype_name='float16',
        num_gpus=None,
        print_device_map=False,
):
    if not use_model_parallel or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return model

    from accelerate import dispatch_model, infer_auto_device_map

    target_dtype = resolve_torch_dtype(dtype_name)
    visible_gpu_count = torch.cuda.device_count()
    if num_gpus is not None:
        visible_gpu_count = min(max(int(num_gpus), 1), visible_gpu_count)
    if visible_gpu_count < 2:
        return model
    model = model.cpu()
    if target_dtype != torch.float32:
        model = model.to(dtype=target_dtype)

    max_memory = {
        idx: f"{max(int(torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3) - 2, 10)}GiB"
        for idx in range(visible_gpu_count)

    }
    max_memory['cpu'] = '120GiB'

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=['LlamaDecoderLayer'],
        dtype=target_dtype,
    )
    if print_device_map:
        _print_device_map_summary(device_map)
    return dispatch_model(model, device_map=device_map)


# evaluates accuracy
def evaluate_cliphead(
        model, loader, class_vectors, remap_class_idxs=None,
        return_confusion=False, task_info=None, return_loss=False, silent=False):
    """Evaluate a model with a cliphead on a dataset."""
    model.eval()
    correct = 0
    total = 0

    totals = np.array([0] * class_vectors.shape[0])
    corrects = np.array([0] * class_vectors.shape[0])

    device = get_device(model)
    losses = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating CLIP head model', disable=silent):
            encodings = model(inputs.to(device))
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)

            if task_info is not None:
                task_map = task_info['task_map']
                data_label_task = task_map[labels].to(device)
                task_features = torch.stack(task_info['task_features'], dim=0).transpose(-1, -2)[data_label_task]
                outputs = torch.einsum('ij,ijk->ik', normed_encodings, task_features)
                remap_class_idxs = task_info['remap_class_idxs']
            else:
                outputs = normed_encodings @ class_vectors.T
            pred = outputs.argmax(dim=1)
            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels]
            else:
                remapped_labels = labels
            loss = loss_fn(outputs, remapped_labels.to(device))
            losses += [loss.item()]

            for gt, p in zip(labels, pred):
                if remap_class_idxs is not None:
                    idx = gt
                    gt = remap_class_idxs[gt]
                else:
                    idx = gt

                is_correct = (gt == p).item()
                correct += is_correct

                if return_confusion:
                    totals[idx] += 1
                    corrects[idx] += is_correct

            total += encodings.shape[0]

    overall_loss = np.mean(losses)

    if return_confusion:
        return correct / sum(totals), list(map(lambda a: a[0] / a[1], zip(corrects, totals)))
    else:
        if return_loss:
            return correct / total, overall_loss
        return correct / total

def prepare_data(config, device='cuda'):
    """ Load all dataloaders required for experiment. """
    if isinstance(config, list):
        return [prepare_data(c, device) for c in config]

    dataset_name = config['name']

    import dataset.configs as config_module
    data_config = deepcopy(getattr(config_module, dataset_name))
    data_config.update(config)
    data_config['device'] = device
    data_config['num_workers'] = min(os.cpu_count(), data_config['num_workers'])
    env_batch_size = None
    env_num_workers = None
    if data_config['type'] in {'snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail'}:
        env_batch_size = os.getenv('NLI_BATCH_SIZE')
        env_num_workers = os.getenv('NLI_NUM_WORKERS')
    if env_batch_size is not None:
        data_config['batch_size'] = int(env_batch_size)
    if env_num_workers is not None:
        data_config['num_workers'] = min(os.cpu_count(), int(env_num_workers))

    # NLI datasets
    if data_config['type'] == 'snli':
        from dataset.snli import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'mnli':
        from dataset.mnli import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'sick':
        from dataset.sick import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'qnli':
        from dataset.qnli import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'rte':
        from dataset.rte import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'scitail':
        from dataset.scitail import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    # -------------------------------------------------

    elif data_config['type'] == 'eurosat':
        from dataset.eurosat import prepare_test_loaders, prepare_train_loaders
        print('Loading Eurosat')
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'stanford_cars':
        print('Loading Cars')
        from dataset.cars import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'dtd':
        print('Loading DTD')
        from dataset.dtd import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'mnist':
        print('Loading MNIST')
        from dataset.mnist import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'gtsrb':
        print('Loading GTSRB')
        from dataset.gtsrb import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'svhn':
        print('Loading SVHN')
        from dataset.svhn import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'sun397':
        print('Loading SUN397')
        from dataset.sun397 import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'resisc45':
        print('Loading RESISC45')
        from dataset.resisc45 import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    else:
        raise NotImplementedError(config['type'])

    return {
        'train': train_loaders,
        'test': test_loaders,
    }


def validate_vision_lora_state_dict(
    state_dict,
    *,
    checkpoint_path,
    expected_rank=None,
    expected_target_modules=None,
):
    """Validate a repository-format vision LoRA before allocating merge state."""
    layers = defaultdict(dict)
    for key, tensor in state_dict.items():
        for marker, component in (
            ('.lora_A.default.weight', 'A'),
            ('.lora_B.default.weight', 'B'),
            ('.lora_A.weight', 'A'),
            ('.lora_B.weight', 'B'),
        ):
            if key.endswith(marker):
                layers[key[:-len(marker)]][component] = tensor
                break
    if not layers:
        raise ValueError(
            f"Vision checkpoint contains no LoRA A/B tensors: {checkpoint_path}"
        )

    target_modules = set(expected_target_modules or ())
    errors = []
    modules_by_parent = defaultdict(set)
    for layer, pair in sorted(layers.items()):
        missing = {'A', 'B'} - set(pair)
        if missing:
            errors.append(f"{layer}: missing {sorted(missing)}")
            continue
        A, B = pair['A'], pair['B']
        if A.ndim != 2 or B.ndim != 2:
            errors.append(
                f"{layer}: expected matrices, got A{tuple(A.shape)} B{tuple(B.shape)}"
            )
            continue
        rank = int(A.shape[0])
        if int(B.shape[1]) != rank:
            errors.append(
                f"{layer}: inconsistent rank, got A{tuple(A.shape)} B{tuple(B.shape)}"
            )
        if expected_rank is not None and rank != int(expected_rank):
            errors.append(
                f"{layer}: expected rank {expected_rank}, got rank {rank}"
            )
        module_name = layer.rsplit('.', 1)[-1]
        parent_name = layer.rsplit('.', 1)[0]
        modules_by_parent[parent_name].add(module_name)
        if target_modules and module_name not in target_modules:
            errors.append(
                f"{layer}: unexpected target module {module_name!r}"
            )
    if target_modules:
        for parent_name, observed_modules in sorted(modules_by_parent.items()):
            # CLIP/SigLIP encoder self-attention blocks must contain the full
            # q/k/v/out projection set. SigLIP also has a pooling
            # ``head.attention.out_proj`` that legitimately matches PEFT's
            # ``out_proj`` target without exposing separate q/k/v modules.
            is_encoder_attention = (
                '.encoder.layers.' in parent_name
                and parent_name.endswith('.self_attn')
            )
            if is_encoder_attention and observed_modules != target_modules:
                errors.append(
                    f"{parent_name}: expected target modules "
                    f"{sorted(target_modules)}, got {sorted(observed_modules)}"
                )
    if errors:
        raise ValueError(
            f"Invalid vision LoRA checkpoint {checkpoint_path}:\n"
            + "\n".join(errors)
        )
    return frozenset(layers)


def _vision_lora_layer_dimensions(state_dict):
    """Return each validated LoRA layer's (input, output) dimensions."""
    layers = defaultdict(dict)
    for key, tensor in state_dict.items():
        for marker, component in (
            ('.lora_A.default.weight', 'A'),
            ('.lora_B.default.weight', 'B'),
            ('.lora_A.weight', 'A'),
            ('.lora_B.weight', 'B'),
        ):
            if key.endswith(marker):
                layers[key[:-len(marker)]][component] = tensor
                break
    return {
        layer: (int(pair['A'].shape[1]), int(pair['B'].shape[0]))
        for layer, pair in layers.items()
        if set(pair) == {'A', 'B'}
        and pair['A'].ndim == 2
        and pair['B'].ndim == 2
    }


def prepare_param_handler(ft_config):
    """Load FT model parameter extractors"""
    if ft_config.get('type', None) == 'lora':
        from ft_handlers import LoRAHandler
        return LoRAHandler
    raise ValueError('PGSB experiments require LoRA checkpoints.')


def prepare_llama(config, device):
    """Load the Llama-3-8B NLI sequence-classification adapters."""
    import psutil
    process = psutil.Process()

    bases = []
    peft_config = LoraConfig(task_type=config["peft_config"]["task_type"],
                             inference_mode=config["peft_config"]["inference_mode"],
                             r=config["peft_config"]["r"],
                             lora_alpha=config["peft_config"]["lora_alpha"],
                             lora_dropout=config["peft_config"]["lora_dropout"],
                             target_modules=config["peft_config"]["target_modules"],
    )
    model_name_or_path = config['name']
    load_dtype_name = str(
        config.get(
            'load_dtype',
            os.getenv(
                'LLAMA_LOAD_DTYPE',
                'float16',
            ),
        )
    ).lower()
    load_dtype = resolve_torch_dtype(load_dtype_name)
    print(f"loading llama with dtype={load_dtype_name}")

    print("memory before loading ", process.memory_info().rss / 1024**2)
    model_loader = AutoModelForSequenceClassification.from_pretrained
    model_kwargs = {
        'return_dict': True,
        'cache_dir': config['cachedir'],
        'low_cpu_mem_usage': True,
        'torch_dtype': load_dtype,
    }
    model_kwargs['num_labels'] = 3
    if config.get('local_files_only', False):
        model_kwargs['local_files_only'] = True
    ptm_model = hf_load_with_local_fallback(
        model_loader,
        model_name_or_path,
        **model_kwargs,
    )

    adapter_cache_dir = config.get(
        'adapter_cachedir',
        config.get('cachedir'),
    )
    local_files_only = bool(config.get('local_files_only', False))
    for base_path in tqdm(config['bases'], desc="Preparing Models"):
        bases.append(
            load_lora_state_dict(
                base_path,
                base_model=ptm_model,
                cache_dir=adapter_cache_dir,
                local_files_only=local_files_only,
            )
        )

    if model_name_or_path == 'meta-llama/Meta-Llama-3-8B':
        ptm_model_path = 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_pretrained_model'
        base_model = hf_load_with_local_fallback(
            lambda model_id, **kwargs: PeftModel.from_pretrained(model=ptm_model, model_id=model_id, **kwargs),
            ptm_model_path,
            **_hf_download_kwargs(
                cache_dir=adapter_cache_dir,
                local_files_only=local_files_only,
            ),
        )
    else:
        base_model = get_peft_model(ptm_model, peft_config)  # Load LoRA model
    print("memory after loading ", process.memory_info().rss / 1024**2)
    return {
        'bases': bases,
        'new': base_model,
        'cache_signature': build_model_cache_signature(
            model_name_or_path,
            config['bases'],
            config.get('peft_config', config.get('ft_config')),
        ),
    }


def _prepare_hf_vision(config, device, model_builder):
    """Load local or Hub PEFT vision adapters with a model-specific builder.

    Native PEFT adapters are read directly with ``load_peft_weights``. This
    avoids repeatedly injecting adapters into a shared model and keeps Hub and
    repository-format checkpoints on the same validation path.
    """
    bases = []
    ranks = config['ft_config']['r']
    rank_list = list(ranks) if isinstance(ranks, (list, tuple)) else None
    alpha = config['ft_config'].get('lora_alpha')
    if rank_list is not None:
        if len(rank_list) != len(config['bases']):
            raise ValueError(
                "Mixed-rank vision config requires one rank per checkpoint: "
                f"got {len(rank_list)} ranks and {len(config['bases'])} checkpoints."
            )
        if not isinstance(alpha, (list, tuple)) or len(alpha) != len(rank_list):
            raise ValueError(
                "Mixed-rank vision config requires lora_alpha to be a list "
                "with the same length as r and bases."
            )

    reference_layers = None
    reference_dimensions = None
    reference_checkpoint = None
    expected_targets = config['ft_config'].get('target_modules', ())
    cache_dir = config.get(
        'adapter_cachedir',
        config.get('cachedir'),
    )
    local_files_only = bool(config.get('local_files_only', False))
    for i, raw_base_path in enumerate(config['bases']):
        base_path = str(raw_base_path)
        is_repository_checkpoint = base_path.endswith('.pt')
        expected_rank = rank_list[i] if rank_list is not None else ranks
        expected_alpha = (
            alpha[i] if rank_list is not None else alpha
        )

        if is_repository_checkpoint:
            checkpoint_path = Path(base_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Vision LoRA checkpoint does not exist: {base_path}"
                )
            lora_state_dict = torch.load(
                checkpoint_path,
                map_location='cpu',
            )
        else:
            adapter_config, lora_state_dict = _load_peft_adapter_artifacts(
                base_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            _validate_peft_vision_adapter_config(
                adapter_config,
                adapter_source=base_path,
                expected_rank=expected_rank,
                expected_alpha=expected_alpha,
                expected_target_modules=expected_targets,
                expected_base_model=config.get('base_type'),
            )

        lora_state_dict = _normalize_vision_lora_state_dict(
            lora_state_dict
        )
        checkpoint_layers = validate_vision_lora_state_dict(
            lora_state_dict,
            checkpoint_path=base_path,
            expected_rank=expected_rank,
            expected_target_modules=expected_targets,
        )
        checkpoint_dimensions = _vision_lora_layer_dimensions(
            lora_state_dict
        )
        if reference_layers is None:
            reference_layers = checkpoint_layers
            reference_dimensions = checkpoint_dimensions
            reference_checkpoint = base_path
        elif checkpoint_layers != reference_layers:
            missing = sorted(reference_layers - checkpoint_layers)
            extra = sorted(checkpoint_layers - reference_layers)
            raise ValueError(
                f"Vision LoRA layer set differs between {reference_checkpoint} "
                f"and {base_path}; missing={missing[:3]}, extra={extra[:3]}"
            )
        elif checkpoint_dimensions != reference_dimensions:
            mismatched = [
                layer
                for layer in sorted(reference_layers)
                if checkpoint_dimensions[layer] != reference_dimensions[layer]
            ]
            layer = mismatched[0]
            raise ValueError(
                "Vision LoRA layer dimensions differ between "
                f"{reference_checkpoint} and {base_path} at {layer}: "
                f"{reference_dimensions[layer]} vs "
                f"{checkpoint_dimensions[layer]}"
            )
        bases.append(lora_state_dict)
        print(f"Loaded model from {base_path}")

    if rank_list is not None:
        config_i = deepcopy(config)
        max_rank_index = max(range(len(rank_list)), key=rank_list.__getitem__)
        config_i['ft_config']['r'] = rank_list[max_rank_index]
        config_i['ft_config']['lora_alpha'] = alpha[max_rank_index]
        base_model = model_builder(config=config_i, device=device)
    else:
        base_model = model_builder(config, device)

    carrier_rank = max(rank_list) if rank_list is not None else ranks
    carrier_layers = validate_vision_lora_state_dict(
        base_model.state_dict(),
        checkpoint_path='merge carrier',
        expected_rank=carrier_rank,
        expected_target_modules=expected_targets,
    )
    if reference_layers is not None and carrier_layers != reference_layers:
        missing = sorted(carrier_layers - reference_layers)
        extra = sorted(reference_layers - carrier_layers)
        raise ValueError(
            "Vision LoRA checkpoint layers do not match the merge carrier; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if reference_dimensions is not None:
        carrier_dimensions = _vision_lora_layer_dimensions(
            base_model.state_dict()
        )
        mismatched = [
            layer
            for layer in sorted(reference_layers)
            if carrier_dimensions[layer] != reference_dimensions[layer]
        ]
        if mismatched:
            layer = mismatched[0]
            raise ValueError(
                "Vision LoRA checkpoint dimensions do not match the merge "
                f"carrier at {layer}: checkpoint "
                f"{reference_dimensions[layer]} vs carrier "
                f"{carrier_dimensions[layer]}"
            )

    return {
        'bases': bases,
        'new': base_model,
        'cache_signature': build_model_cache_signature(
            config['base_type'],
            config['bases'],
            config.get('ft_config'),
        ),
    }


def prepare_hf_clip(config, device):
    """Load Hugging Face CLIP vision models from config."""
    return _prepare_hf_vision(config, device, get_hf_clip_model_from_config)


def prepare_hf_siglip(config, device):
    """Load Hugging Face SigLIP vision models from config."""
    from models.huggingface_siglip import (
        get_model_from_config as get_hf_siglip_model_from_config,
    )

    return _prepare_hf_vision(config, device, get_hf_siglip_model_from_config)


def prepare_models(config, device='cuda'):
    """ Load all pretrained models in config. """
    if config['name'].startswith('hf_siglip'):
        return prepare_hf_siglip(config, device)
    elif config['name'].startswith('hf_clip'):
        return prepare_hf_clip(config, device)
    elif (
        config.get('family') == 'llama'
        or config['name'].lower().startswith('meta-llama')
    ):
        return prepare_llama(config, device)
    else:
        raise NotImplementedError(config['name'])


def prepare_experiment_config(config):
    """ Load all functions/classes/models requested in config to experiment config dict. """
    models = prepare_models(config['model'], device=config['device'])
    if models.get('cache_signature') is not None and config['task_merge_config'].get('cache_namespace') is None:
        config['task_merge_config']['cache_namespace'] = models['cache_signature']

    # Get preprocessors from base model
    # We can do that as they are the same for all datasets -- see HFLoRACLIPVisionModel
    if len(models['bases']) > 0 and hasattr(models['new'], 'train_preprocess'):
        if isinstance(config['dataset'], list):
            for dataset_config in config['dataset']:
                dataset_config['train_preprocess'] = models['new'].train_preprocess
                dataset_config['eval_preprocess'] = models['new'].val_preprocess
        else:
            config['dataset']['train_preprocess'] = models['new'].train_preprocess
            config['dataset']['eval_preprocess'] = models['new'].val_preprocess

    data = prepare_data(config['dataset'], device=config['device'])
    new_config = {
        'data': data,
        'models': models,
        'task_merge_config': config['task_merge_config'],
        'param_handler': prepare_param_handler(config['model'].get('ft_config', defaultdict()))
    }
    # Add outstanding elements
    for key in config:
        if key not in new_config:
            new_config[key] = config[key]
    return new_config


def get_config_from_name(name, device=None):
    """ Load config based on its name. """
    supplied = Path(name).expanduser()
    if supplied.is_file() or supplied.is_absolute() or supplied.parent != Path('.'):
        p = supplied if supplied.suffix == '.py' else supplied.with_suffix('.py')
    else:
        p = Path('configs', name)
        if p.suffix != '.py':
            p = p.with_suffix('.py')
    if not p.exists():
        raise FileNotFoundError(f"Config file {p} does not exist.")

    if p.parent == Path('configs') and not supplied.is_absolute() and supplied.parent == Path('.'):
        out = __import__('configs', fromlist=[p.stem]).__dict__.get(p.stem)
        if out is None:
            module = __import__(f'configs.{p.stem}', fromlist=[p.stem])
            out = module.config
        else:
            out = out.config
    else:
        module_name = f"_runtime_merge_config_{hashlib.sha1(str(p.resolve()).encode()).hexdigest()}"
        spec = importlib.util.spec_from_file_location(module_name, p.resolve())
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load config module from {p}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = module.config
    out = deepcopy(out)
    out['task_merge_config'] = canonicalize_merge_config(
        out.get('task_merge_config', {})
    )
    if device is None and 'device' not in out:
        out['device'] = 'cuda'
    elif device is not None:
        out['device'] = device
    return out


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate PGSB paper merges')
    parser.add_argument('--config', type=str, required=True, help='Config name or Python file')
    parser.add_argument('--representation', choices=('matrix_per_layer',))
    parser.add_argument('--merge_space', type=canonicalize_merge_space,
                        choices=('full', 'knots', 'core', 'pgsb'))
    parser.add_argument('--merge_method', choices=('tv', 'ties', 'dare-ties', 'tsv', 'cart', 'iso-c'))
    parser.add_argument('--scaling_coeffs', type=float, help='Task scaling coefficient')
    parser.add_argument('--scaling_early_stop_floor', type=float, default=None,
                        help='Do not stop the scaling search below this value')
    parser.add_argument('--isotropize', type=str2bool, nargs='?', const=True, default=None,
                        help='Apply Iso-C to the merged update')
    parser.add_argument('--topK', type=int, help='[TIES] Percentage retained')
    parser.add_argument('--dare_pruning_coeffs', type=float, help='[DARE-TIES] Drop probability')
    parser.add_argument('--cart_pruning_rank', type=float, help='[CART] Residual rank fraction')
    parser.add_argument('--cart_scaling_coeffs', type=float,
                        help='[CART] Centered residual scaling coefficient')
    parser.add_argument('--disk_cache', type=str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--disk_cache_dir', type=str, default=None)
    parser.add_argument('--cache_namespace', type=str, default=None)
    parser.add_argument('--pgsb_rank_mode', choices=('exact', 'energy'), default=None)
    parser.add_argument('--pgsb_energy', type=float, default=None)
    parser.add_argument('--pgsb_eps', type=float, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--model_parallel', type=str2bool, nargs='?', const=True, default=None,
                        help='[NLI] Shard evaluation across visible GPUs')
    parser.add_argument('--model_parallel_dtype', type=str, default=None,
                        choices=('float16', 'fp16', 'half', 'bfloat16', 'bf16', 'float32', 'fp32'))
    parser.add_argument('--model_parallel_num_gpus', type=int, default=None)
    parser.add_argument('--model_parallel_print_device_map', type=str2bool,
                        nargs='?', const=True, default=None)
    parser.add_argument('--wandb', type=int, default=0, choices=(0, 1))
    parser.add_argument('--wandb_project', type=str, default='PGSB')
    parser.add_argument('--wandb_entity', type=str)
    return parser.parse_args()


def merge_args_into_task_merge_config(task_merge_config, args):
    """
    Overwrite or add entries in task_merge_config with non-None values from args (argparse.Namespace).
    """
    for key, value in vars(args).items():
        if key in {
            "config",
            "output_json",
        }:
            continue  # Don't overwrite config name in task_merge_config
        if value is not None:
            task_merge_config[key] = value
    normalized = canonicalize_merge_config(task_merge_config)
    task_merge_config.clear()
    task_merge_config.update(normalized)
    return task_merge_config


##########################################################################################################################
#################################################### HELPER FUNCTIONS ####################################################
##########################################################################################################################

def set_seed(seed):
    """Set the seed for reproducibility."""
    #print("Setting Seed to {}".format(seed))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


def get_clip_encodings(path):
    return torch.load(path)
