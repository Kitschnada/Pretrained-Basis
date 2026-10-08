import json
from pathlib import Path

import numpy as np


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


def build_search_setup(task_merge_config):
    scaling_search = task_merge_config.get("search_scaling_coeffs", np.arange(0.1, 10, step=0.1))
    # Keep the shared paper-search initialization explicit and independently
    # overridable. The scalar merge config is mutated during search, while
    # these values define the starting point used by coordinate search.
    default_params = {
        "scaling_coeffs": task_merge_config.get(
            "linearsearch_initial_scaling_coeffs",
            0.1,
        ),
        "topK": task_merge_config.get(
            "linearsearch_initial_topK",
            10,
        ),
        "cart_pruning_rank": task_merge_config.get("cart_pruning_rank", 0.04),
        "cart_scaling_coeffs": task_merge_config.get("cart_scaling_coeffs", 0.1),
        "dare_pruning_coeffs": task_merge_config.get(
            "linearsearch_initial_dare_pruning_coeffs",
            1e-5,
        ),
    }
    order_of_processing_params = ["scaling_coeffs"]
    search_config = {
        "scaling_coeffs": np.array(scaling_search),
        "topK": np.array(task_merge_config.get("search_topK", (np.arange(1, 11, step=1) * 10)[::-1])),
        "dare_pruning_coeffs": task_merge_config.get(
            "search_dare_pruning_coeffs",
            [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-5][::-1],
        ),
        "cart_pruning_rank": task_merge_config.get("search_cart_pruning_rank", [0.04, 0.08, 0.16, 0.32]),
        "cart_scaling_coeffs": np.array(task_merge_config.get("search_cart_scaling_coeffs", scaling_search)),
    }
    merge_method = task_merge_config.get("merge_method", "")
    if "dare" in merge_method:
        order_of_processing_params.append("dare_pruning_coeffs")
    if "ties" in merge_method:
        order_of_processing_params.append("topK")
    if "cart" in merge_method:
        order_of_processing_params.append("cart_pruning_rank")
        order_of_processing_params.append("cart_scaling_coeffs")
    return default_params, order_of_processing_params, search_config


def write_output_json(output_json, payload):
    if not output_json:
        return
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

class SearchHistory:
    """Persist each validation candidate without mixing in final-test metrics."""

    def __init__(self, *, output_json, search_order, search_config):
        self.search_order = list(search_order)
        self.search_grid = {
            name: _json_safe(list(search_config[name]))
            for name in self.search_order
        }
        self.history = []
        self.jsonl_path = (
            Path(output_json).with_suffix(".search.jsonl")
            if output_json else None
        )
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("", encoding="utf-8")

    def append_candidate(self, record):
        safe_record = _json_safe({
            "candidate_index": len(self.history),
            **record,
        })
        self.history.append(safe_record)
        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
                handle.flush()

    def summary(self, *, selected_parameters):
        return _json_safe({
            "scope": "validation parameter search; final test evaluation is excluded",
            "search_order": self.search_order,
            "search_grid": self.search_grid,
            "num_candidates_planned": sum(map(len, self.search_grid.values())),
            "num_candidates_evaluated": len(self.history),
            "selected_parameters": selected_parameters,
            "history_jsonl": str(self.jsonl_path) if self.jsonl_path else None,
        })
