import os
import sys
from pathlib import Path
from copy import deepcopy
import time
import gc

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_scripts.vision_linearsearch_common import write_output_json
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import transformers

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def run_BIG_function(args):

    
    # Get config
    config_name = args.config
    print("Config name : ", config_name)

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

    task_heads = torch.load(raw_config.get('task_heads_path', 'heads.pt'), map_location='cpu')

    finetuned_llama3_8b = {
        'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122, 'qnli': 94.48512585812358, 'rte': 89.85507246376812, 'scitail': 96.51928504233303, }
    fine_tuned_acc = raw_config.get('finetuned_accuracies', finetuned_llama3_8b) or {}
    print(f'Finetuned Accs: {fine_tuned_acc}')

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
        print(f"Time taken to merge: {time.time() - t0}")

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
        if ingredients_path is None or not os.path.exists(ingredients_path):
            Merge.transform(config['task_merge_config'])

        test_result = merge_and_eval(Merge, EVAL_SPLIT='test', instance_params=config['task_merge_config'])
        datasets = ['snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail']
        normalized_values = [test_result.get(name + '_norm_acc') for name in datasets]
        normalized_values.append(test_result.get('Average_norm_acc'))
        test_results = " & ".join(
            "N/A" if value is None else str(np.round(value, 2))
            for value in normalized_values
        ) + " \\\\"
        print(f"Normalized Test results: {test_results}")
        print(test_result)
        write_output_json(args.output_json, {
            'args': vars(args),
            'seed': BIGSEED,
            'test_result': test_result,
            'normalized_test_results': test_results,
        })


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
