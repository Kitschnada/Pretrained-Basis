import os
from copy import deepcopy
import json
import time
import gc
from pathlib import Path

import numpy as np
import torch

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_scripts.vision_linearsearch_common import SearchHistory
from task_merger import get_merge_handler
from utils import (
    evaluate_logits,
    get_config_from_name,
    merge_args_into_task_merge_config,
    parse_eval_args,
    prepare_experiment_config,
    prepare_model_for_parallel_eval,
    set_seed,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _json_safe(value.tolist())
    return value

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import transformers

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def build_search_setup(task_merge_config):
    scaling_search = task_merge_config.get('search_scaling_coeffs')
    if scaling_search is None:
        scaling_search = np.arange(0.1, 10.0, step=0.1)

    default_params = {
        'scaling_coeffs': task_merge_config.get('scaling_coeffs', 0.3),
        'topK': task_merge_config.get('topK', 10),
        'cart_pruning_rank': task_merge_config.get('cart_pruning_rank', 0.04),
        'cart_scaling_coeffs': task_merge_config.get('cart_scaling_coeffs', 0.1),
        'dare_pruning_coeffs': task_merge_config.get('dare_pruning_coeffs', 0.1),
    }
    order_of_processing_params = ['scaling_coeffs']
    search_config = {
        'scaling_coeffs': np.array(scaling_search),
        'topK': np.array(task_merge_config.get('search_topK', (np.arange(1, 11, step=1) * 10))),
        'dare_pruning_coeffs': task_merge_config.get(
            'search_dare_pruning_coeffs',
            [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-5][::-1],
        ),
        'cart_pruning_rank': task_merge_config.get('search_cart_pruning_rank', [0.04, 0.08, 0.16, 0.32]),
        'cart_scaling_coeffs': np.array(task_merge_config.get('search_cart_scaling_coeffs', scaling_search)),
    }
    merge_method = task_merge_config.get('merge_method', '')
    if 'dare' in merge_method:
        order_of_processing_params.append('dare_pruning_coeffs')
    if 'ties' in merge_method:
        order_of_processing_params.append('topK')
    if 'cart' in merge_method:
        order_of_processing_params.append('cart_pruning_rank')
        order_of_processing_params.append('cart_scaling_coeffs')
    return default_params, order_of_processing_params, search_config


def run_BIG_function(args):
    EVAL_SPLIT = 'val'
    EVAL_TEST = True

    
    # Get config
    config_name = args.config
    print("Config name : ", config_name)
    EARLY_STOPPING_STEPS = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    raw_config = get_config_from_name(config_name, device=device)
    BIGSEED = int(raw_config.get('seed', 420))
    print('Seed:', BIGSEED)
    set_seed(BIGSEED)
    print(raw_config['task_merge_config'])
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])
    print(f"mask_class labels: {mask_class}")


    # Parameters are tuned in the order specified in search_config
    default_params, order_of_processing_params, search_config = build_search_setup(config['task_merge_config'])
    search_recorder = SearchHistory(
        output_json=getattr(args, 'output_json', None),
        search_order=order_of_processing_params,
        search_config=search_config,
    )
    print(f"default params: {default_params}")
    print(f"order_of_processing_params: {order_of_processing_params}")

    task_heads = torch.load(raw_config.get('task_heads_path', 'heads.pt'), map_location='cpu')

    finetuned_llama3_8b = {
        'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122, 'qnli': 94.48512585812358, 'rte': 89.85507246376812, 'scitail': 96.51928504233303, }


    print("Using Llama fine-tuned acc")
    fine_tuned_acc = raw_config.get('finetuned_accuracies', finetuned_llama3_8b) or {}
    has_complete_baselines = all(
        fine_tuned_acc.get(name) is not None and float(fine_tuned_acc[name]) > 0
        for name in dataset_names
    )
    search_metric = 'Average_norm_acc' if has_complete_baselines else 'Average_acc'

    print(f'Finetuned Accs: {fine_tuned_acc}')
    print(search_config)

    def merge_and_eval(Merge, EVAL_SPLIT='val', instance_params=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        print('Creating Merge')

        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)
        t0 = time.time()
        merged_model = Merge.merge(config['task_merge_config'])
        reported_merge_sec = time.time() - t0
        print(f"Time taken to merge: {reported_merge_sec}")

        merged_model.config.pad_token_id = 128001
        merged_model.config.use_cache = False
        merged_model.config.pretraining_tp = 1
        merged_model = prepare_model_for_parallel_eval(
            merged_model,
            use_model_parallel=config['task_merge_config'].get('model_parallel', False),
            dtype_name=config['task_merge_config'].get('model_parallel_dtype', 'float16'),
            num_gpus=config['task_merge_config'].get('model_parallel_num_gpus'),
            print_device_map=config['task_merge_config'].get('model_parallel_print_device_map', False),
        )

        print('Evaluate Merged Model on Each Dataset')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        normalized_count = 0
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    # Inject task head into model
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]].to(device=param.device, dtype=param.dtype))

            acc = evaluate_logits(merged_model, loader, device, mask_class[i])
            baseline = fine_tuned_acc.get(dataset_names[i])
            if baseline is not None and float(baseline) > 0:
                normalized = (acc * 100) / float(baseline) * 100
                all_results[dataset_names[i] + '_norm_acc'] = normalized
                avg_norm_accuracy += normalized
                normalized_count += 1
                print(f"{dataset_names[i]} Normalized accuracy is {np.round(normalized, 3)}")
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            avg_accuracy += acc * 100
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy = avg_norm_accuracy / normalized_count if normalized_count else None
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        if avg_norm_accuracy is not None:
            print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return all_results


    with torch.no_grad():
        lora_state_dicts = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            lora_state_dicts,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )

        ingredients_path = config['task_merge_config'].get('ingredients_path')
        transform_performed = ingredients_path is None or not os.path.exists(ingredients_path)
        if transform_performed:
            Merge.transform(config['task_merge_config'])

        print(config['task_merge_config'])
        scaling_early_stop_floor = config['task_merge_config'].get('scaling_early_stop_floor')
        scaling_early_stop_floor = None if scaling_early_stop_floor is None else float(scaling_early_stop_floor)
        for param in order_of_processing_params:
            early_stopping = EARLY_STOPPING_STEPS
            best_val_results = {search_metric: 0.0}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] = value
                all_results = merge_and_eval(
                    Merge,
                    EVAL_SPLIT=EVAL_SPLIT,
                    instance_params=instance_params,
                )
                is_best = (
                    all_results[search_metric]
                    >= best_val_results[search_metric]
                )
                stopped_after_candidate = False
                skipped_early_stop_decrement = False
                if is_best:
                    best_val_results = deepcopy(all_results)
                    early_stopping = EARLY_STOPPING_STEPS
                else:
                    if (
                        param == 'scaling_coeffs'
                        and scaling_early_stop_floor is not None
                        and float(value) < scaling_early_stop_floor
                    ):
                        skipped_early_stop_decrement = True
                    else:
                        early_stopping -= 1
                        if early_stopping <= 0:
                            stopped_after_candidate = True
                metric_keys = {'Average_acc', 'Average_norm_acc'}
                metric_keys.update(str(name) for name in dataset_names)
                metric_keys.update(
                    str(name) + '_norm_acc' for name in dataset_names
                )
                search_recorder.append_candidate(
                    {
                        'split': EVAL_SPLIT,
                        'searched_parameter': param,
                        'candidate_value': value,
                        'parameters': instance_params,
                        'search_metric': search_metric,
                        'search_metric_value': all_results[search_metric],
                        'is_best': is_best,
                        'early_stopping_remaining': early_stopping,
                        'early_stop_decrement_skipped': (
                            skipped_early_stop_decrement
                        ),
                        'stopped_after_candidate': stopped_after_candidate,
                        'metrics': {
                            key: all_results[key]
                            for key in metric_keys
                            if key in all_results
                        },
                    }
                )
                if stopped_after_candidate:
                    print("Early stopping")
                    break
            default_params[param] = best_val_results[param]
        search_summary = search_recorder.summary(
            selected_parameters=default_params
        )

        if EVAL_TEST:
            # Evaluate on the test set with the best topK and scaling co-efficient
            instance_params = deepcopy(default_params)
            print("Best params :", instance_params)
            test_result = merge_and_eval(
                Merge,
                EVAL_SPLIT='test',
                instance_params=instance_params,
            )
            datasets = ['snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail']
            normalized_values = [test_result.get(name + '_norm_acc') for name in datasets]
            normalized_values.append(test_result.get('Average_norm_acc'))
            test_results = " & ".join(
                "N/A" if value is None else str(np.round(value, 2))
                for value in normalized_values
            ) + " \\\\"
            print(f"Normalized Test results: {test_results}")
            print(test_result)
            if args.output_json:
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(
                        _json_safe({
                            'args': vars(args),
                            'seed': BIGSEED,
                            'search_metric': search_metric,
                            'has_complete_baselines': has_complete_baselines,
                            'best_parameters': instance_params,
                            'test_result': test_result,
                            'normalized_test_results': test_results,
                            'search_summary': search_summary,
                        }),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding='utf-8',
                )


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
