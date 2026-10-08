import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import evaluate_cliphead, get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def run_BIG_function(args):
    EVAL_SPLIT = 'test'
    # Get config
    CONFIG_NAME = args.config

    print("Running with config: ", CONFIG_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    BIGSEED = int(raw_config.get('seed', 420))
    print('Seed:', BIGSEED)
    set_seed(BIGSEED)
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = [i['name'] for i in raw_config['dataset']]
    dataloaders = [i for i in config['data']]

    model_type = config['model']['base_type']
    rank = config['model']['ft_config'].get('r', None)
    peft_type = config['model']['ft_config'].get('type')
    fine_tuned_acc = raw_config.get('finetuned_accuracies')
    if fine_tuned_acc is None:
        fine_tuned_acc = get_vision_accuracies(
            model_type,
            peft_type=peft_type,
            rank=rank,
            dataset_names=dataset_names,
        )
    fine_tuned_acc = fine_tuned_acc or {}

    print(raw_config['task_merge_config'])
    with torch.no_grad():
        all_results = deepcopy(config['task_merge_config'])
        print('Creating Merge')
        # iniitalize merging function
        models = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            models,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        Merge.transform(config['task_merge_config'])
        # set task scaling coefficients
        if 'scaling_coeffs' not in config['task_merge_config']:
            raise KeyError(
                "Missing 'scaling_coeffs' in task_merge_config. "
                "The fixed vision preset should provide this field before calling 8vision_pertask.py."
            )
        Merge.set_scaling_coeffs(config['task_merge_config']['scaling_coeffs'])
        merged_model = Merge.merge(config['task_merge_config'])
        print('Evaluate Merged Model on Each Dataset')
        print("Using config: ", config['task_merge_config'])
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        normalized_count = 0
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
            baseline = fine_tuned_acc.get(dataset_names[i])
            normalized = None
            if baseline is not None and float(baseline) > 0:
                normalized = (acc * 100) / float(baseline) * 100
                print(f"{dataset_names[i]} Normalized accuracy is {np.round(normalized, 3)}")
                all_results[dataset_names[i] + '_norm_acc'] = normalized
                avg_norm_accuracy += normalized
                normalized_count += 1
            else:
                print(
                    f"{dataset_names[i]} Normalized accuracy is unavailable "
                    "(no single-task baseline supplied)"
                )
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            avg_accuracy += acc * 100
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy = (
            avg_norm_accuracy / normalized_count if normalized_count else None
        )

        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        if avg_norm_accuracy is not None:
            print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        else:
            print('Average Normalized Accuracy is unavailable')
        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        datasets = ['stanford_cars', 'dtd', 'eurosat', 'gtsrb', 'mnist', 'resisc45', 'sun397', 'svhn']
        normalized_values = [
            all_results.get(dataset + '_norm_acc')
            for dataset in datasets
        ]
        test_results = " & ".join(
            "N/A" if value is None else f"{np.round(value, 2)}"
            for value in normalized_values
        )
        test_results += (
            " & N/A \\\\"
            if all_results['Average_norm_acc'] is None
            else f" & {np.round(all_results['Average_norm_acc'], 2)} \\\\"
        )
        print(f"Normalized Test results: {test_results}")
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    _json_safe({
                        'args': vars(args),
                        'seed': BIGSEED,
                        'test_result': all_results,
                        'normalized_test_results': test_results,
                    }),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
        print('Finished!')


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
