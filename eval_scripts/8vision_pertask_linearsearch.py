import time
import sys
from copy import deepcopy

import numpy as np
import torch
try:
    import wandb
except ImportError:
    wandb = None

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import (
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
    merge_args_into_task_merge_config,
    parse_eval_args,
    prepare_experiment_config,
    set_seed,
)
from eval_scripts.vision_linearsearch_common import (
    SearchHistory, build_search_setup, write_output_json,
)


def resolve_finetuned_accuracies(
    raw_config,
    *,
    model_type,
    peft_type,
    rank,
    dataset_names,
):
    configured = raw_config.get('finetuned_accuracies')
    legacy_tasks = raw_config.get('legacy_finetuned_accuracy_tasks')
    if legacy_tasks is None:
        if configured is not None:
            return dict(configured)
        legacy = get_vision_accuracies(
            model_type,
            peft_type=peft_type,
            rank=rank,
            dataset_names=dataset_names,
        )
        return legacy or {}

    if isinstance(legacy_tasks, (str, bytes)) or not isinstance(
        legacy_tasks, (list, tuple, set)
    ):
        raise ValueError(
            "legacy_finetuned_accuracy_tasks must be a sequence of task names"
        )

    dataset_names = list(dataset_names)
    unknown_tasks = set(legacy_tasks) - set(dataset_names)
    if unknown_tasks:
        raise ValueError(
            "legacy_finetuned_accuracy_tasks contains tasks outside the "
            f"runtime dataset: {sorted(unknown_tasks)}"
        )

    resolved = dict(configured or {})
    legacy_task_set = set(legacy_tasks)
    tasks_to_fill = [
        task
        for task in dataset_names
        if task in legacy_task_set and task not in resolved
    ]
    if not tasks_to_fill:
        return resolved

    selected_rank = rank
    if isinstance(rank, (list, tuple)):
        if len(rank) != len(dataset_names):
            raise ValueError(
                "A rank list must contain one entry per runtime dataset: "
                f"got {len(rank)} ranks and {len(dataset_names)} datasets"
            )
        selected_rank = [
            rank[index]
            for index, task in enumerate(dataset_names)
            if task in tasks_to_fill
        ]

    legacy = get_vision_accuracies(
        model_type,
        peft_type=peft_type,
        rank=selected_rank,
        dataset_names=tasks_to_fill,
    )
    for task in tasks_to_fill:
        if legacy is not None and task in legacy:
            resolved[task] = legacy[task]
    return resolved


def select_search_metric(
    task_merge_config, *, has_complete_baselines
):
    if not has_complete_baselines:
        return 'Average_acc'
    return 'Average_norm_acc'


def result_text_path(output_json):
    if output_json:
        return Path(output_json).with_suffix('.txt')
    return Path('results.txt')


def run_BIG_function(args):
    EVAL_TEST = True
    # EVAL_SPLIT = 'test'
    EVAL_SPLIT = 'val'


    # Get config
    config_name = args.config
    print("Config name : ", config_name)
    EARLY_STOPPING_STEPS = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    BIGSEED = int(raw_config.get('seed', 420))
    print('Seed:', BIGSEED)
    set_seed(BIGSEED)
    if args.wandb and wandb is None:
        raise ImportError("wandb is not installed, but --wandb=1 was requested.")
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f"{config_name}_{args.merge_method}_{args.merge_space}_{args.representation}",
                   config={'raw_config': raw_config, 'args': vars(args)})
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = [i['name'] for i in raw_config['dataset']]
    dataloaders = [i for i in config['data']]

    default_params, order_of_processing_params, search_config = build_search_setup(config['task_merge_config'])
    search_recorder = SearchHistory(
        output_json=getattr(args, 'output_json', None),
        search_order=order_of_processing_params,
        search_config=search_config,
    )

    model_type = config['model']['base_type']
    rank = config['model']['ft_config'].get('r', None)
    peft_type = config['model']['ft_config'].get('type')
    fine_tuned_acc = resolve_finetuned_accuracies(
        raw_config,
        model_type=model_type,
        peft_type=peft_type,
        rank=rank,
        dataset_names=dataset_names,
    )
    has_complete_baselines = all(
        fine_tuned_acc.get(dataset_name) is not None
        and float(fine_tuned_acc[dataset_name]) > 0
        for dataset_name in dataset_names
    )

    print(f'Finetuned Accs: {fine_tuned_acc}')

    def merge_and_eval(merger, EVAL_SPLIT='val', instance_config=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_config}')
        all_results = deepcopy(instance_config)
        # initialize merging function
        print('Creating Merge')
        t1 = time.time()
        merger.transform(instance_config)
        # set task scaling coefficients
        merger.set_scaling_coeffs(instance_config['scaling_coeffs'])
        merged_model = merger.merge(instance_config)
        t2 = time.time()
        reported_merge_sec = t2 - t1
        print(f'Merging time: {reported_merge_sec:.2f} seconds')

        print('Evaluate Merged Model on Each Dataset')
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
            avg_norm_accuracy / normalized_count
            if normalized_count
            else None
        )
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        if avg_norm_accuracy is not None:
            print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        else:
            print('Average Normalized Accuracy is unavailable')

        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        # Log the merge evaluation results to wandb
        if args.wandb:
            wandb.log({**all_results, "params": instance_config})
        return all_results

    with torch.no_grad():
        print(search_config)
        models = np.array([i for i in config['models']['bases']])

        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        merger = MergeClass(
            deepcopy(models),
            pretrained_model=deepcopy(config['models']['new']),
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        print(config['task_merge_config'])
        scaling_early_stop_floor = config['task_merge_config'].get('scaling_early_stop_floor')
        scaling_early_stop_floor = None if scaling_early_stop_floor is None else float(scaling_early_stop_floor)
        search_metric = select_search_metric(
            config['task_merge_config'],
            has_complete_baselines=has_complete_baselines,
        )
        for param in order_of_processing_params:
            early_stopping = EARLY_STOPPING_STEPS
            best_val_results = {search_metric: float('-inf')}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] = value
                config['task_merge_config'].update(instance_params)
                all_results = merge_and_eval(
                    merger=merger,
                    EVAL_SPLIT=EVAL_SPLIT,
                    instance_config=config['task_merge_config']
                )
                is_best = (
                    all_results.get(search_metric, 0)
                    >= best_val_results.get(search_metric, 0)
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
                        if early_stopping == 0:
                            stopped_after_candidate = True
                metric_keys = {
                    'Average_acc',
                    'Average_norm_acc',
                }
                metric_keys.update(dataset_names)
                metric_keys.update(
                    dataset_name + '_norm_acc'
                    for dataset_name in dataset_names
                )
                search_recorder.append_candidate(
                    {
                        'split': EVAL_SPLIT,
                        'searched_parameter': param,
                        'candidate_value': value,
                        'parameters': instance_params,
                        'search_metric': search_metric,
                        'search_metric_value': all_results.get(search_metric),
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
            config['task_merge_config'].update(instance_params)
            test_result = merge_and_eval(
                merger=merger,
                EVAL_SPLIT='test',
                instance_config=config['task_merge_config']
            )
            datasets = ['stanford_cars', 'dtd', 'eurosat', 'gtsrb', 'mnist', 'resisc45', 'sun397', 'svhn']
            normalized_values = [
                test_result.get(dataset + '_norm_acc')
                for dataset in datasets
            ]
            test_results = " & ".join(
                "N/A" if value is None else f"{np.round(value, 2):.2f}"
                for value in normalized_values
            )
            average_normalized = test_result.get('Average_norm_acc')
            test_results += (
                " & N/A \\\\"
                if average_normalized is None
                else f" & {np.round(average_normalized, 2):.2f} \\\\"
            )
            print(f"Normalized Test results: {test_results}")
            print(test_result)
            summary_path = result_text_path(args.output_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(f"Args: {vars(args)}\n")
                f.write(f"Search metric: {search_metric}\n")
                f.write(f"Normalized Test results: {test_results}\n")
                f.write(f"Test result dict: {test_result}\n")
                f.write(f"Best parameters: {instance_params}\n\n")
            if args.output_json:
                write_output_json(
                    args.output_json,
                    {
                        'args': vars(args),
                        'seed': BIGSEED,
                        'search_metric': search_metric,
                        'has_complete_baselines': has_complete_baselines,
                        'best_parameters': instance_params,
                        'test_result': test_result,
                        'normalized_test_results': test_results,
                        'search_summary': search_summary,
                    },
                )
            # Log final test results to wandb
            if args.wandb:

                wandb.log({"final_test": test_result, "best_parameters": instance_params})
    if args.wandb:
        # Finish the wandb run
        wandb.finish()


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
