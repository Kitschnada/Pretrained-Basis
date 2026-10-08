"""Layerwise LoRA merging in Full, KnOTS, Core Space and PGSB coordinates.

The PGSB implementation preserves the main-experiment support construction,
pretrained-coupled rotations, tie breaking and float64 operations. Iso-C is an
optional post-processing step applied to any merge operator in its merge space.
"""

import gc
import hashlib
import math
import os
from copy import deepcopy
from pathlib import Path

import einops
import torch
from torch import nn
from tqdm import tqdm

from merge_naming import canonicalize_merge_config
from merging_functions import ties_merging


class TaskMerger(nn.Module):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__()

        self.device = device
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models))
        self.param_handler = param_handler
        self.finetuned_models = finetuned_models
        self.ftms_params = [param_handler(ft_model) for ft_model in finetuned_models]
        self.pretrained_model = pretrained_model.cpu()
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config


    def randbin(self, M, N, P, device=None, dtype=torch.float32):
        P = 1 - P
        return torch.randint(2, size=(M, N), device=device, dtype=dtype).bernoulli(P)


    def _get_merge_compute_device(self):
        if torch.cuda.is_available():
            if isinstance(self.device, int):
                return torch.device(f'cuda:{self.device}')
            if isinstance(self.device, str) and self.device.startswith('cuda'):
                return torch.device(self.device)
            return torch.device('cuda')
        return torch.device('cpu')


    def _to_merge_compute(self, tensor, device, dtype=torch.float32):
        return tensor.to(device=device, dtype=dtype)


    def transform(self, *args, **kwargs):
        return



class MatrixPerLayerMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.cache = {}


    def set_scaling_coeffs(self, scaling_coeffs):
        self.scaling_coeffs = torch.tensor(scaling_coeffs)


    def _get_disk_cache_dir(self, merge_config):
        if merge_config.get('disk_cache', True) is False:
            return None
        cache_root = merge_config.get('disk_cache_dir') or os.getenv('MERGE_DISK_CACHE_DIR') or 'data/merge_cache'
        cache_namespace = merge_config.get('cache_namespace', 'default')
        cache_dir = Path(cache_root) / cache_namespace / 'matrix_per_layer'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


    def _get_disk_cache_path(self, merge_config, cache_key):
        cache_dir = self._get_disk_cache_dir(merge_config)
        if cache_dir is None or cache_key is None:
            return None
        hashed_key = hashlib.sha1(cache_key.encode('utf-8')).hexdigest()
        return cache_dir / f'{hashed_key}.pt'


    def _load_cached_tensor(self, cache_path):
        payload = torch.load(cache_path, map_location='cpu')
        if isinstance(payload, dict) and 'tensor' in payload:
            return payload['tensor']
        return payload


    def _save_cached_tensor(self, cache_path, tensor, cache_key):
        torch.save(
            {
                'tensor': tensor.detach().cpu(),
                'dtype': str(tensor.dtype),
                'cache_key': cache_key,
            },
            cache_path,
        )


    def _apply_delta(self, param_dict, key, delta_w):
        param = param_dict.get(key)
        if param is None:
            return
        param.data += self.scaling_coeffs * delta_w.type_as(param.data)


    def _merge_method_mutates_inputs(self, merge_method):
        return merge_method in {'dare-ties'}


    def _prepare_tensors_for_merge(self, tensors, device, merge_method):
        if isinstance(tensors, torch.Tensor):
            prepared = tensors.to(device)
            if self._merge_method_mutates_inputs(merge_method):
                prepared = prepared.clone()
            return prepared
        prepared = [tensor.to(device) for tensor in tensors]
        if self._merge_method_mutates_inputs(merge_method):
            prepared = [tensor.clone() for tensor in prepared]
        return prepared


    def _process_ties(self, tensor_list, topK=10):
        if isinstance(tensor_list, torch.Tensor):
            original_shape = tuple(tensor_list.shape[1:])
            flat_tensors = tensor_list.reshape(tensor_list.shape[0], -1)
            merged_tv, rows_to_keep, mask = ties_merging(flat_tensors, topK=topK)
            return merged_tv.reshape(original_shape)

        original_shape = tensor_list[0].shape
        tensor_list = list(map(torch.flatten, tensor_list))
        merged_tv, rows_to_keep, mask = ties_merging(tensor_list, topK=topK)
        return merged_tv.reshape(original_shape)


    def get_tsv_delta_w(self, ftms_task_dirs, merge_config=None, cache_key=None):
        merge_config = merge_config or {}
        if isinstance(ftms_task_dirs, torch.Tensor):
            target_device = ftms_task_dirs.device
            target_dtype = ftms_task_dirs.dtype
            num_tasks = int(ftms_task_dirs.shape[0])
            target_rows = int(ftms_task_dirs.shape[1])
            target_cols = int(ftms_task_dirs.shape[2])
        else:
            target_device = ftms_task_dirs[0].device
            target_dtype = ftms_task_dirs[0].dtype
            num_tasks = len(ftms_task_dirs)
            target_rows = int(ftms_task_dirs[0].shape[0])
            target_cols = int(ftms_task_dirs[0].shape[1])

        mem_cache_key = None
        if cache_key is not None:
            mem_cache_key = (
                f"tsv:v2:{cache_key}:shape={target_rows}x{target_cols}:tasks={num_tasks}"
            )

        if mem_cache_key is not None and mem_cache_key in self.cache:
            cached_tensor = self.cache[mem_cache_key]
            if tuple(cached_tensor.shape) == (target_rows, target_cols):
                return cached_tensor.to(device=target_device, dtype=target_dtype)

        disk_cache_path = self._get_disk_cache_path(merge_config, mem_cache_key)
        if disk_cache_path is not None and disk_cache_path.exists():
            cached_tensor = self._load_cached_tensor(disk_cache_path)
            if tuple(cached_tensor.shape) == (target_rows, target_cols):
                if mem_cache_key is not None:
                    self.cache[mem_cache_key] = cached_tensor
                return cached_tensor.to(device=target_device, dtype=target_dtype)

        matrix_stack = ftms_task_dirs if isinstance(ftms_task_dirs, torch.Tensor) else torch.stack(ftms_task_dirs, dim=0)
        matrix_stack64 = matrix_stack.to(torch.float64)
        u, s, vh = torch.linalg.svd(matrix_stack64, full_matrices=False)
        reduced_index_s = max(1, int(s.shape[-1] * (1 / max(1, num_tasks))))
        reduced_index_s = min(reduced_index_s, s.shape[-1])
        sum_u = u[:, :, :reduced_index_s].permute(1, 0, 2).reshape(matrix_stack.shape[1], num_tasks * reduced_index_s)
        sum_s = s[:, :reduced_index_s].reshape(num_tasks * reduced_index_s)
        sum_v = vh[:, :reduced_index_s, :].reshape(num_tasks * reduced_index_s, matrix_stack.shape[2])
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        merged_tensor = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).detach().cpu()

        if mem_cache_key is not None:
            self.cache[mem_cache_key] = merged_tensor
        if disk_cache_path is not None:
            self._save_cached_tensor(disk_cache_path, merged_tensor, mem_cache_key)

        return merged_tensor.to(device=target_device, dtype=target_dtype)


    def get_core_matrices(self, ftms_params_ab, key, merge_config):
        if key in self.cache:
            return self.cache[key]
        # Extract A and B matrices from all tasks
        A_list, B_list = zip(*[ftm[key] for ftm in ftms_params_ab])
        r, n = A_list[0].shape
        m, _ = B_list[0].shape

        A_stack = torch.cat(A_list, dim=0)  # shape: (T*r, n)
        B_stack = torch.cat(B_list, dim=1)  # shape: (m, T*r)

        Vh_A_ref = torch.linalg.svd(A_stack.to(torch.float64), full_matrices=False)[2]  # shape: (T*r, n)
        U_B_ref = torch.linalg.svd(B_stack.to(torch.float64), full_matrices=False)[0]  # shape: (m, T*r)

        M_list = []
        for i, (A, B) in enumerate(zip(A_list, B_list)):
            U_A, S_A, Vh_A = torch.linalg.svd(A.to(torch.float64), full_matrices=False)  # shape: (r, r), (r,), (r, n)
            U_B, S_B, Vh_B = torch.linalg.svd(B.to(torch.float64), full_matrices=False)  # shape: (m, r), (r,), (r, r)

            Q_A = Vh_A @ Vh_A_ref.T
            R_B = U_B_ref.T @ U_B

            # Middle core matrix M = Σ_B * V_B^T * U_A * Σ_A
            M = torch.diag(S_B) @ (Vh_B @ U_A) @ torch.diag(S_A)  # shape: (r, r)
            # Apply alignment to M
            M_aligned = R_B @ M @ Q_A  # shape: (T*r, T*r)

            M_list.append(M_aligned)

        self.cache[key] = (M_list, U_B_ref, Vh_A_ref)
        return M_list, U_B_ref, Vh_A_ref


    def get_dare_delta_w(self, M_list, dare_coeff=0.3, merge=True):
        if isinstance(M_list, torch.Tensor):
            masked = M_list.clone()
            for i in range(masked.shape[0]):
                m = self.randbin(
                    masked[i].shape[0],
                    masked[i].shape[1],
                    dare_coeff,
                    device=masked[i].device,
                    dtype=masked[i].dtype,
                )
                masked[i] = masked[i] * m
                masked[i] = masked[i] / (1 - dare_coeff)
            return masked.sum(dim=0) if merge else masked

        for i in range(len(M_list)):
            m = self.randbin(
                M_list[i].shape[0],
                M_list[i].shape[1],
                dare_coeff,
                device=M_list[i].device,
                dtype=M_list[i].dtype,
            )
            M_list[i] = M_list[i] * m
            M_list[i] = M_list[i] / (1 - dare_coeff)
        return torch.stack(M_list).sum(dim=0) if merge else torch.stack(M_list)


    def get_cart_delta_w(self, ftms_task_dirs, pruning_rank=0.04, scaling_coeffs=1.):
        matrix_stack = ftms_task_dirs if isinstance(ftms_task_dirs, torch.Tensor) else torch.stack(ftms_task_dirs, dim=0)
        theta_avg = matrix_stack.mean(dim=0)
        tau_stack = matrix_stack.to(torch.float64) - theta_avg.to(torch.float64).unsqueeze(0)
        U, S, Vh = torch.linalg.svd(tau_stack, full_matrices=False)
        pruning_rank_k = min(max(1, math.ceil(pruning_rank * S.shape[-1])), S.shape[-1])
        truncated = (U[:, :, :pruning_rank_k] * S[:, :pruning_rank_k].unsqueeze(1)) @ Vh[:, :pruning_rank_k, :]
        return theta_avg + scaling_coeffs * truncated.sum(dim=0).to(dtype=theta_avg.dtype, device=theta_avg.device)


    def get_knots_components(self, ftms_task_dirs, merge_config=None, cache_key=None):
        merge_config = merge_config or {}
        mem_cache_key = cache_key
        if mem_cache_key is not None and mem_cache_key in self.cache:
            return self.cache[mem_cache_key]

        disk_cache_path = self._get_disk_cache_path(merge_config, mem_cache_key)
        if disk_cache_path is not None and disk_cache_path.exists():
            payload = torch.load(disk_cache_path, map_location='cpu')
            cached_value = (payload['U'], payload['S'], payload['Vs'])
            if mem_cache_key is not None:
                self.cache[mem_cache_key] = cached_value
            return cached_value

        if ftms_task_dirs is None:
            raise ValueError(f"KnOTS cache miss for {cache_key}, but no tensors were provided to rebuild it.")

        compute_device = self._get_merge_compute_device()
        compute_dtype = torch.float32
        stack = torch.cat([self._to_merge_compute(tensor, compute_device, compute_dtype) for tensor in ftms_task_dirs], dim=1)
        U, S, Vh = torch.linalg.svd(stack, full_matrices=False)
        # Keep only supported basis components
        U = U[:, S > 1e-5].type(torch.float32)
        Vh = Vh[S > 1e-5].type(torch.float32)
        S = S[S > 1e-5].type(torch.float32)

        S[S <= 1e-5] = 0
        Vs = list(einops.rearrange(Vh, 'Tr (b c) -> b Tr c', b=len(ftms_task_dirs)))
        cached_value = (U.cpu(), S.cpu(), [v.cpu() for v in Vs])

        if mem_cache_key is not None:
            self.cache[mem_cache_key] = cached_value
        if disk_cache_path is not None:
            torch.save(
                {
                    'U': cached_value[0],
                    'S': cached_value[1],
                    'Vs': cached_value[2],
                    'cache_key': mem_cache_key,
                },
                disk_cache_path,
            )
        return cached_value


    def _resolve_pretrained_key(self, key, pretrained_sd):
        if key in pretrained_sd:
            return key
        parts = key.rsplit('.', 1)
        alt_key = parts[0] + '.base_layer.' + parts[1] if len(parts) == 2 else key
        if alt_key in pretrained_sd:
            return alt_key
        raise KeyError(
            f"Cannot find pretrained weight for '{key}'. "
            f"Tried '{key}' and '{alt_key}'. "
            f"Available keys (first 5): {list(pretrained_sd.keys())[:5]}"
        )


    def _choose_energy_rank_from_singulars(self, singular_values, energy_ratio, eps=1e-8):
        if singular_values.numel() == 0:
            return 0
        positive = int((singular_values > eps).sum().item())
        if positive <= 0:
            return 0
        energy = singular_values[:positive] ** 2
        total = energy.sum().item()
        if total <= eps:
            return positive
        target = min(max(float(energy_ratio), 0.0), 1.0)
        cumsum = torch.cumsum(energy, dim=0) / total
        return int((cumsum >= target).float().argmax().item()) + 1


    def _orthogonal_basis_from_slices(self, slices, rank_mode='exact', energy_ratio=0.99, eps=1e-8):
        valid_slices = [tensor for tensor in slices if tensor is not None and tensor.numel() > 0 and tensor.shape[1] > 0]
        if not valid_slices:
            if slices:
                prototype = slices[0]
                return prototype.new_zeros((prototype.shape[0], 0)), prototype.new_zeros((0,))
            return torch.zeros((0, 0), dtype=torch.float64), torch.zeros((0,), dtype=torch.float64)

        stack = torch.cat(valid_slices, dim=1)
        U, S, _ = torch.linalg.svd(stack, full_matrices=False)
        positive = int((S > eps).sum().item())
        if positive <= 0:
            return U[:, :0], S[:0]

        rank_mode = str(rank_mode).lower()
        if rank_mode == 'exact':
            keep = positive
        elif rank_mode == 'energy':
            keep = self._choose_energy_rank_from_singulars(S, energy_ratio, eps=eps)
        else:
            raise ValueError(f"Unknown rank_mode: {rank_mode}")

        keep = min(max(int(keep), 0), positive)
        return U[:, :keep], S[:keep]


    def _get_pretrained_config(self):
        candidates = [getattr(self.pretrained_model, 'config', None)]
        for attr in ('model', 'base_model', 'vision_model'):
            obj = getattr(self.pretrained_model, attr, None)
            if obj is None:
                continue
            candidates.append(getattr(obj, 'config', None))
            candidates.append(getattr(getattr(obj, 'model', None), 'config', None))
            candidates.append(getattr(getattr(obj, 'base_model', None), 'config', None))

        for config in candidates:
            if config is None:
                continue
            if hasattr(config, 'vision_config') and getattr(config, 'vision_config') is not None:
                return config.vision_config
            return config
        return None


    def _make_block_ranges(self, total_dim, num_blocks):
        if num_blocks is None or int(num_blocks) <= 1:
            return [(0, total_dim)]
        num_blocks = int(num_blocks)
        if total_dim % num_blocks != 0:
            return [(0, total_dim)]
        block_size = total_dim // num_blocks
        return [(idx * block_size, (idx + 1) * block_size) for idx in range(num_blocks)]


    def _repeat_head_rows(self, matrix, src_heads, dst_heads):
        src_heads = int(src_heads)
        dst_heads = int(dst_heads)
        if src_heads <= 0 or dst_heads <= 0:
            return None
        if src_heads == dst_heads:
            return matrix
        if matrix.shape[0] % src_heads != 0 or dst_heads % src_heads != 0:
            return None
        head_dim = matrix.shape[0] // src_heads
        repeat = dst_heads // src_heads
        reshaped = matrix.reshape(src_heads, head_dim, matrix.shape[1])
        return reshaped.repeat_interleave(repeat, dim=0).reshape(dst_heads * head_dim, matrix.shape[1])


    def _average_head_rows(self, matrix, src_heads, dst_heads):
        src_heads = int(src_heads)
        dst_heads = int(dst_heads)
        if src_heads <= 0 or dst_heads <= 0:
            return None
        if src_heads == dst_heads:
            return matrix
        if matrix.shape[0] % src_heads != 0 or src_heads % dst_heads != 0:
            return None
        head_dim = matrix.shape[0] // src_heads
        group = src_heads // dst_heads
        reshaped = matrix.reshape(dst_heads, group, head_dim, matrix.shape[1])
        return reshaped.mean(dim=1).reshape(dst_heads * head_dim, matrix.shape[1])


    def _average_head_cols(self, matrix, src_heads, dst_heads):
        src_heads = int(src_heads)
        dst_heads = int(dst_heads)
        if src_heads <= 0 or dst_heads <= 0:
            return None
        if src_heads == dst_heads:
            return matrix
        if matrix.shape[1] % src_heads != 0 or src_heads % dst_heads != 0:
            return None
        head_dim = matrix.shape[1] // src_heads
        group = src_heads // dst_heads
        reshaped = matrix.reshape(matrix.shape[0], dst_heads, group, head_dim)
        return reshaped.mean(dim=2).reshape(matrix.shape[0], dst_heads * head_dim)


    def _orient_basis_columns(self, basis):
        if basis.numel() == 0 or basis.shape[1] == 0:
            return basis
        basis = basis.clone()
        for col in range(basis.shape[1]):
            vector = basis[:, col]
            if vector.numel() == 0:
                continue
            max_idx = int(torch.argmax(torch.abs(vector)).item())
            if vector[max_idx] < 0:
                basis[:, col] = -vector
        return basis


    def compact_lora_svd(self, A, B, eps=1e-8, compute_device=None, compute_dtype=torch.float64):
        if compute_device is None:
            compute_device = self._get_merge_compute_device()

        A64 = self._to_merge_compute(A, compute_device, compute_dtype)
        B64 = self._to_merge_compute(B, compute_device, compute_dtype)
        Q_b, R_b = torch.linalg.qr(B64, mode='reduced')
        Q_a, R_a = torch.linalg.qr(A64.T, mode='reduced')
        compact = R_b @ R_a.T
        U_hat, S, Vh_hat = torch.linalg.svd(compact, full_matrices=False)
        keep = S > eps
        if int(keep.sum().item()) == 0:
            return B64[:, :0], S[:0], A64.T[:, :0]

        U = Q_b @ U_hat[:, keep]
        V = Q_a @ Vh_hat[keep].T
        return U, S[keep], V


    def compact_lora_svd_batch(self, A_list, B_list, eps=1e-8, compute_device=None, compute_dtype=torch.float64):
        if compute_device is None:
            compute_device = self._get_merge_compute_device()
        if len(A_list) != len(B_list):
            raise ValueError("A_list and B_list must have the same length.")
        if not A_list:
            return [], [], []

        A_stack = torch.stack([self._to_merge_compute(A, compute_device, compute_dtype) for A in A_list], dim=0)
        B_stack = torch.stack([self._to_merge_compute(B, compute_device, compute_dtype) for B in B_list], dim=0)
        Q_b, R_b = torch.linalg.qr(B_stack, mode='reduced')
        Q_a, R_a = torch.linalg.qr(A_stack.transpose(-1, -2), mode='reduced')
        compact = R_b @ R_a.transpose(-1, -2)
        U_hat, S, Vh_hat = torch.linalg.svd(compact, full_matrices=False)

        U_list, S_list, V_list = [], [], []
        for idx in range(A_stack.shape[0]):
            keep = S[idx] > eps
            if int(keep.sum().item()) == 0:
                U_list.append(B_stack[idx, :, :0])
                S_list.append(S[idx, :0])
                V_list.append(A_stack[idx].transpose(0, 1)[:, :0])
                continue
            U_list.append(Q_b[idx] @ U_hat[idx, :, keep])
            S_list.append(S[idx, keep])
            V_list.append(Q_a[idx] @ Vh_hat[idx, keep].transpose(0, 1))
        return U_list, S_list, V_list


    def _get_pgsb_compute_dtype(self):
        # PGSB key path uses float64 on both CPU/GPU to reduce numerical
        # perturbations that can inflate effective support/rank.
        return torch.float64


    def _normalize_merge_cache_value(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return tuple(value.detach().cpu().reshape(-1).tolist())
        if isinstance(value, dict):
            return tuple((key, self._normalize_merge_cache_value(val)) for key, val in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self._normalize_merge_cache_value(val) for val in value)
        if hasattr(value, 'tolist') and not isinstance(value, (str, bytes)):
            return self._normalize_merge_cache_value(value.tolist())
        if isinstance(value, (Path, torch.dtype, torch.device)):
            return str(value)
        return value


    def _get_merge_result_cache_key(self, cache_key, merge_config):
        if cache_key is None:
            return None
        excluded = {
            'disk_cache',
            'disk_cache_dir',
            'cache_namespace',
            'ingredients_path',
            'log_merge_ranks',
            'model_parallel',
            'model_parallel_dtype',
            'model_parallel_num_gpus',
            'model_parallel_print_device_map',
            'scaling_coeffs',
        }
        payload = []
        for key, value in sorted(merge_config.items()):
            if key in excluded or key.startswith('search_'):
                continue
            payload.append((key, self._normalize_merge_cache_value(value)))
        return f"merged:{cache_key}:{repr(payload)}"


    def infer_attention_block_spec(self, key, weight_shape):
        m, n = weight_shape
        key_lower = key.lower()
        config = self._get_pretrained_config()
        hidden_size = getattr(config, 'hidden_size', None)
        num_attention_heads = getattr(config, 'num_attention_heads', None)
        num_key_value_heads = getattr(config, 'num_key_value_heads', None)

        spec = {
            'family': 'generic',
            'module_type': 'other',
            'left_mode': 'global',
            'right_mode': 'global',
            'left_ranges': [(0, m)],
            'right_ranges': [(0, n)],
        }

        if hidden_size == 768 and num_attention_heads == 12 and m == n == 768:
            spec['family'] = 'vit'
            if any(name in key_lower for name in ('q_proj', 'k_proj', 'v_proj')):
                spec['module_type'] = 'qkv'
                spec['left_mode'] = 'block'
                spec['left_ranges'] = self._make_block_ranges(m, num_attention_heads)
            elif 'out_proj' in key_lower:
                spec['module_type'] = 'out_proj'
                spec['right_mode'] = 'block'
                spec['right_ranges'] = self._make_block_ranges(n, num_attention_heads)
            return spec

        if hidden_size is None or num_attention_heads is None:
            if m == n == 768:
                spec['family'] = 'vit'
                if any(name in key_lower for name in ('q_proj', 'k_proj', 'v_proj')):
                    spec['module_type'] = 'qkv'
                    spec['left_mode'] = 'block'
                    spec['left_ranges'] = self._make_block_ranges(m, 12)
                elif 'out_proj' in key_lower:
                    spec['module_type'] = 'out_proj'
                    spec['right_mode'] = 'block'
                    spec['right_ranges'] = self._make_block_ranges(n, 12)
            return spec

        query_group_heads = num_attention_heads
        kv_heads = num_key_value_heads or num_attention_heads
        if (
            kv_heads is not None
            and num_attention_heads is not None
            and int(kv_heads) > 0
            and int(kv_heads) < int(num_attention_heads)
            and hidden_size % int(kv_heads) == 0
        ):
            # For GQA models such as LLaMA-3-8B, q/o directions are more stably
            # organized by KV groups than by individual query heads. Each KV head
            # serves a group of query heads, so we align q/o at that group level.
            query_group_heads = int(kv_heads)

        if 'q_proj' in key_lower and m == hidden_size:
            spec['family'] = 'llama'
            spec['module_type'] = 'q_proj'
            spec['left_mode'] = 'block'
            spec['left_ranges'] = self._make_block_ranges(m, query_group_heads)
        elif any(name in key_lower for name in ('k_proj', 'v_proj')):
            if kv_heads and m % kv_heads == 0:
                spec['family'] = 'llama'
                spec['module_type'] = 'kv_proj'
                spec['left_mode'] = 'block'
                spec['left_ranges'] = self._make_block_ranges(m, kv_heads)
        elif 'o_proj' in key_lower and n == hidden_size:
            spec['family'] = 'llama'
            spec['module_type'] = 'o_proj'
            spec['right_mode'] = 'block'
            spec['right_ranges'] = self._make_block_ranges(n, query_group_heads)

        return spec


    def canonicalize_degenerate_eigenspaces(self, eigenvalues, eigenvectors, tie_break=None, eps=1e-8):
        if eigenvectors.numel() == 0 or eigenvectors.shape[1] <= 1:
            return self._orient_basis_columns(eigenvectors)

        canonical = eigenvectors.clone()
        scale = max(float(eigenvalues.abs().max().item()), 1.0) if eigenvalues.numel() > 0 else 1.0
        if tie_break is not None and tie_break.numel() > 0:
            tie_break = 0.5 * (tie_break + tie_break.T)
            scale = max(scale, float(tie_break.abs().max().item()), 1.0)
        tol = max(float(eps), scale * max(1, eigenvalues.numel()) * float(eps))

        start = 0
        while start < eigenvalues.numel():
            end = start + 1
            while end < eigenvalues.numel() and abs((eigenvalues[end] - eigenvalues[start]).item()) <= tol:
                end += 1
            if end - start > 1 and tie_break is not None and tie_break.numel() > 0:
                group_basis = canonical[:, start:end]
                tie_local = group_basis.T @ tie_break @ group_basis
                tie_local = 0.5 * (tie_local + tie_local.T)
                _, tie_vectors = torch.linalg.eigh(tie_local)
                canonical[:, start:end] = group_basis @ torch.flip(tie_vectors, dims=[1])
            start = end

        return self._orient_basis_columns(canonical)


    def _pgsb_stack_tensors(self, core_list):
        if isinstance(core_list, torch.Tensor):
            return core_list
        if not core_list:
            return torch.zeros((0, 0, 0), dtype=torch.float64)
        return torch.stack(core_list, dim=0)


    def _pgsb_unstack_tensors(self, core_stack, as_list=True):
        if not as_list:
            return core_stack
        if core_stack.numel() == 0:
            return []
        return [core_stack[idx] for idx in range(core_stack.shape[0])]


    def _choose_energy_rank_from_eigenvalues(self, eigenvalues, energy_ratio, eps=1e-8):
        if eigenvalues.numel() == 0:
            return 0
        positive = eigenvalues > eps
        if not torch.any(positive):
            return 0
        active = torch.clamp(eigenvalues[positive], min=0.0)
        total = active.sum().item()
        if total <= eps:
            return int(active.numel())
        target = min(max(float(energy_ratio), 0.0), 1.0)
        cumsum = torch.cumsum(active, dim=0) / total
        return int((cumsum >= target).float().argmax().item()) + 1


    def build_pgsb_consensus_basis(
        self,
        factors,
        full_dim,
        block_ranges,
        prior_matrix=None,
        side='left',
        merge_config=None,
    ):
        merge_config = merge_config or {}
        rank_mode = str(merge_config.get('pgsb_rank_mode', 'exact')).lower()
        energy_ratio = float(merge_config.get('pgsb_energy', 0.99))
        eps = float(merge_config.get('pgsb_eps', 1e-8))

        basis_blocks = []
        block_meta = []
        for start, end in block_ranges:
            local_task_factors = [factor[start:end, :] for factor in factors]

            local_union, _ = self._orthogonal_basis_from_slices(
                local_task_factors,
                rank_mode='exact',
                energy_ratio=1.0,
                eps=eps,
            )
            if local_union.shape[1] == 0:
                continue

            reduced = torch.zeros(
                (local_union.shape[1], local_union.shape[1]),
                dtype=local_union.dtype,
                device=local_union.device,
            )
            for local_factor in local_task_factors:
                if local_factor.shape[1] == 0:
                    continue
                projection = local_union.T @ local_factor
                reduced += projection @ projection.T
            reduced = 0.5 * (reduced + reduced.T)
            eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
            eigenvalues = torch.flip(eigenvalues, dims=[0])
            eigenvectors = torch.flip(eigenvectors, dims=[1])

            prior_reduced = None
            if prior_matrix is not None and prior_matrix.numel() > 0:
                if side == 'left':
                    projected_prior = prior_matrix[start:end, :].T @ local_union
                else:
                    projected_prior = prior_matrix[:, start:end] @ local_union
                prior_reduced = projected_prior.T @ projected_prior

            eigenvectors = self.canonicalize_degenerate_eigenspaces(
                eigenvalues,
                eigenvectors,
                tie_break=prior_reduced,
                eps=eps,
            )

            positive = int((eigenvalues > eps).sum().item())
            if rank_mode == 'exact':
                keep = positive
            elif rank_mode == 'energy':
                keep = self._choose_energy_rank_from_eigenvalues(eigenvalues[:positive], energy_ratio, eps=eps)
            else:
                raise ValueError(f"Unknown pgsb_rank_mode: {rank_mode}")

            keep = min(max(int(keep), 0), positive)
            if keep <= 0:
                continue
            local_basis = self._orient_basis_columns(local_union @ eigenvectors[:, :keep])

            ambient_basis = torch.zeros(
                (full_dim, local_basis.shape[1]),
                dtype=local_basis.dtype,
                device=local_basis.device,
            )
            ambient_basis[start:end, :] = local_basis
            basis_blocks.append(ambient_basis)
            block_meta.append(
                {
                    'start': start,
                    'end': end,
                    'rank': local_basis.shape[1],
                    'eigenvalues': eigenvalues[:keep].detach().cpu(),
                }
            )

        if basis_blocks:
            basis = torch.cat(basis_blocks, dim=1)
        else:
            prototype = factors[0]
            basis = prototype.new_zeros((full_dim, 0))

        meta = {
            'side': side,
            'rank': basis.shape[1],
            'num_blocks': len(block_meta),
            'block_ranks': [item['rank'] for item in block_meta],
            'blocks': block_meta,
        }
        return self._orient_basis_columns(basis), meta


    def _get_pgsb_reduced_ranges(self, side_meta):
        ranges = []
        cursor = 0
        for block_meta in side_meta.get('blocks', []):
            rank = int(block_meta.get('rank', 0))
            if rank <= 0:
                continue
            ranges.append((cursor, cursor + rank))
            cursor += rank
        return ranges


    def _orient_pgsb_gauge(self, core_stack, O_left, O_right, prior_core=None, eps=1e-8):
        if core_stack.numel() == 0 or prior_core is None or prior_core.numel() == 0:
            return core_stack, O_left, O_right, {'sign_flips': 0}

        oriented = core_stack.clone()
        prior_rotated = O_left.T @ prior_core @ O_right
        sign_flips = 0
        diag_dim = min(prior_rotated.shape[0], prior_rotated.shape[1])
        for idx in range(diag_dim):
            score = float(prior_rotated[idx, idx].item())
            if abs(score) <= eps:
                score = float(prior_rotated[:, idx].sum().item())
            if abs(score) <= eps:
                score = float(prior_rotated[idx, :].sum().item())
            if score >= 0:
                continue
            O_right[:, idx] = -O_right[:, idx]
            prior_rotated[:, idx] = -prior_rotated[:, idx]
            oriented[:, :, idx] = -oriented[:, :, idx]
            sign_flips += 1
        return oriented, O_left, O_right, {'sign_flips': sign_flips}


    def _psd_sqrt(self, matrix, eps=1e-8):
        if matrix.numel() == 0:
            return matrix
        matrix = 0.5 * (matrix + matrix.T)
        evals, evecs = torch.linalg.eigh(matrix)
        evals = torch.clamp(evals, min=0.0)
        sqrt_evals = torch.where(evals > eps, torch.sqrt(evals), torch.zeros_like(evals))
        return (evecs * sqrt_evals.unsqueeze(0)) @ evecs.T


    def _solve_pgsb_transport_basis(self, operator_matrix, tie_break=None, block_ranges=None, eps=1e-8):
        dim = int(operator_matrix.shape[0])
        if dim == 0:
            return operator_matrix.new_zeros((0, 0))

        if not block_ranges:
            block_ranges = [(0, dim)]

        basis = torch.zeros((dim, dim), dtype=operator_matrix.dtype, device=operator_matrix.device)
        for start, end in block_ranges:
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            local_operator = 0.5 * (
                operator_matrix[start:end, start:end] + operator_matrix[start:end, start:end].T
            )
            local_evals, local_evecs = torch.linalg.eigh(local_operator)
            local_evals = torch.flip(local_evals, dims=[0])
            local_evecs = torch.flip(local_evecs, dims=[1])
            local_tie_break = None
            if tie_break is not None and tie_break.numel() > 0:
                local_tie_break = tie_break[start:end, start:end]
            local_basis = self.canonicalize_degenerate_eigenspaces(
                local_evals,
                local_evecs,
                tie_break=local_tie_break,
                eps=eps,
            )
            basis[start:end, start:end] = local_basis
        return basis


    def _build_llama_attention_coupled_priors(
        self,
        key,
        pretrained_sd,
        W0,
        Q_left_lora,
        Q_right_lora,
        compute_device,
        compute_dtype,
        eps=1e-8,
    ):
        config = self._get_pretrained_config()
        hidden_size = getattr(config, 'hidden_size', None)
        num_attention_heads = getattr(config, 'num_attention_heads', None)
        num_key_value_heads = getattr(config, 'num_key_value_heads', None)
        if (
            hidden_size is None
            or num_attention_heads is None
            or num_key_value_heads is None
            or int(num_key_value_heads) <= 0
            or int(num_key_value_heads) >= int(num_attention_heads)
        ):
            return None

        key_lower = key.lower()
        pair_key = None
        pair_mode = None
        paired = None
        q_heads = int(num_attention_heads)
        kv_heads = int(num_key_value_heads)

        try:
            if 'q_proj' in key_lower:
                pair_key = self._resolve_pretrained_key(key.replace('q_proj', 'k_proj'), pretrained_sd)
                pair_weight = self._to_merge_compute(pretrained_sd[pair_key], compute_device, compute_dtype)
                paired = self._repeat_head_rows(pair_weight, kv_heads, q_heads)
                pair_mode = 'qk_repeat_k'
            elif 'k_proj' in key_lower:
                pair_key = self._resolve_pretrained_key(key.replace('k_proj', 'q_proj'), pretrained_sd)
                pair_weight = self._to_merge_compute(pretrained_sd[pair_key], compute_device, compute_dtype)
                paired = self._average_head_rows(pair_weight, q_heads, kv_heads)
                pair_mode = 'kq_group_q'
            elif 'o_proj' in key_lower:
                pair_key = self._resolve_pretrained_key(key.replace('o_proj', 'v_proj'), pretrained_sd)
                pair_weight = self._to_merge_compute(pretrained_sd[pair_key], compute_device, compute_dtype)
                paired = self._repeat_head_rows(pair_weight, kv_heads, q_heads)
                pair_mode = 'ov_repeat_v'
            elif 'v_proj' in key_lower:
                pair_key = self._resolve_pretrained_key(key.replace('v_proj', 'o_proj'), pretrained_sd)
                pair_weight = self._to_merge_compute(pretrained_sd[pair_key], compute_device, compute_dtype)
                grouped_cols = self._average_head_cols(pair_weight, q_heads, kv_heads)
                paired = None if grouped_cols is None else grouped_cols.T
                pair_mode = 'vo_group_o'
        except KeyError:
            return None

        if paired is None or tuple(paired.shape) != tuple(W0.shape):
            return None

        left_projected = Q_left_lora.T @ W0 @ paired.T
        right_projected = Q_right_lora.T @ W0.T @ paired
        left_prior = left_projected @ left_projected.T
        right_prior = right_projected @ right_projected.T
        left_prior = 0.5 * (left_prior + left_prior.T)
        right_prior = 0.5 * (right_prior + right_prior.T)

        coupling_norm = float(torch.linalg.norm(left_projected).item() + torch.linalg.norm(right_projected).item())
        if coupling_norm <= eps:
            return None

        return {
            'left_prior': left_prior,
            'right_prior': right_prior,
            'pair_key': pair_key,
            'pair_mode': pair_mode,
            'coupling_norm': coupling_norm,
        }


    def build_pgsb_transport_intersection_gauge(
        self,
        core_list,
        transport_matrix,
        left_meta,
        right_meta,
        block_spec,
        attention_priors=None,
        eps=1e-8,
    ):
        as_list = not isinstance(core_list, torch.Tensor)
        core_stack = self._pgsb_stack_tensors(core_list)
        if core_stack.numel() == 0:
            return [], torch.zeros((0, 0)), torch.zeros((0, 0)), {
                'sum_core_norm': 0.0,
                'transport_norm': 0.0,
                'left_operator_norm': 0.0,
                'right_operator_norm': 0.0,
                'gauge_source': 'identity',
                'sign_flips': 0,
            }

        left_dim = core_stack.shape[1]
        right_dim = core_stack.shape[2]
        dtype = core_stack.dtype
        device = core_stack.device
        sum_core = core_stack.sum(dim=0)
        sum_norm = float(torch.linalg.norm(sum_core).item())
        left_tie = 0.5 * (sum_core @ sum_core.T + (sum_core @ sum_core.T).T)
        right_tie = 0.5 * (sum_core.T @ sum_core + (sum_core.T @ sum_core).T)

        if left_dim == 0 or right_dim == 0:
            O_left = torch.eye(left_dim, dtype=dtype, device=device)
            O_right = torch.eye(right_dim, dtype=dtype, device=device)
            return self._pgsb_unstack_tensors(core_stack, as_list=as_list), O_left, O_right, {
                'sum_core_norm': sum_norm,
                'transport_norm': 0.0,
                'left_operator_norm': 0.0,
                'right_operator_norm': 0.0,
                'gauge_source': 'identity',
                'sign_flips': 0,
            }

        gram_left = torch.matmul(core_stack, core_stack.transpose(-1, -2)).sum(dim=0)
        gram_right = torch.matmul(core_stack.transpose(-1, -2), core_stack).sum(dim=0)
        gram_left = 0.5 * (gram_left + gram_left.T)
        gram_right = 0.5 * (gram_right + gram_right.T)
        gram_left_sqrt = self._psd_sqrt(gram_left, eps=eps)
        gram_right_sqrt = self._psd_sqrt(gram_right, eps=eps)

        pair_key = None
        pair_mode = None
        if attention_priors is not None:
            left_prior = attention_priors['left_prior']
            right_prior = attention_priors['right_prior']
            transport_norm = float(attention_priors.get('coupling_norm', 0.0))
            left_operator = gram_left_sqrt @ left_prior @ gram_left_sqrt
            right_operator = gram_right_sqrt @ right_prior @ gram_right_sqrt
            left_operator = 0.5 * (left_operator + left_operator.T)
            right_operator = 0.5 * (right_operator + right_operator.T)
            gauge_source = 'attention_kernel'
            pair_key = attention_priors.get('pair_key')
            pair_mode = attention_priors.get('pair_mode')
            if (
                transport_norm <= eps
                or float(torch.linalg.norm(left_operator).item()) <= eps
                or float(torch.linalg.norm(right_operator).item()) <= eps
            ):
                left_operator = left_tie
                right_operator = right_tie
                gauge_source = 'lora_sum'
        else:
            G = transport_matrix
            transport_norm = float(torch.linalg.norm(G).item())
            gauge_source = 'transport'

            if transport_norm <= eps:
                left_operator = left_tie
                right_operator = right_tie
                gauge_source = 'lora_sum'
            else:
                left_operator = gram_left_sqrt @ G @ gram_right @ G.T @ gram_left_sqrt
                right_operator = gram_right_sqrt @ G.T @ gram_left @ G @ gram_right_sqrt
                left_operator = 0.5 * (left_operator + left_operator.T)
                right_operator = 0.5 * (right_operator + right_operator.T)
                if (
                    float(torch.linalg.norm(left_operator).item()) <= eps
                    or float(torch.linalg.norm(right_operator).item()) <= eps
                ):
                    left_operator = left_tie
                    right_operator = right_tie
                    gauge_source = 'lora_sum'

        left_ranges = None
        right_ranges = None
        if block_spec.get('left_mode') == 'block':
            left_ranges = self._get_pgsb_reduced_ranges(left_meta)
        if block_spec.get('right_mode') == 'block':
            right_ranges = self._get_pgsb_reduced_ranges(right_meta)

        O_left = self._solve_pgsb_transport_basis(
            left_operator,
            tie_break=left_tie,
            block_ranges=left_ranges,
            eps=eps,
        )
        O_right = self._solve_pgsb_transport_basis(
            right_operator,
            tie_break=right_tie,
            block_ranges=right_ranges,
            eps=eps,
        )

        rotated = torch.matmul(O_left.T.unsqueeze(0), core_stack)
        rotated = torch.matmul(rotated, O_right)
        rotated, O_left, O_right, orient_meta = self._orient_pgsb_gauge(
            rotated,
            O_left,
            O_right,
            prior_core=sum_core if sum_norm > eps else None,
            eps=eps,
        )
        return self._pgsb_unstack_tensors(rotated, as_list=as_list), O_left, O_right, {
            'sum_core_norm': sum_norm,
            'transport_norm': transport_norm,
            'left_operator_norm': float(torch.linalg.norm(left_operator).item()),
            'right_operator_norm': float(torch.linalg.norm(right_operator).item()),
            'gauge_source': gauge_source,
            'pair_key': pair_key,
            'pair_mode': pair_mode,
            **orient_meta,
        }


    def get_pgsb_components(self, ftms_params_ab, key, merge_config, pretrained_sd):
        rank_mode = str(merge_config.get('pgsb_rank_mode', 'exact')).lower()
        energy_ratio = float(merge_config.get('pgsb_energy', 0.99))
        eps = float(merge_config.get('pgsb_eps', 1e-8))
        cache_key = f"{key}:pgsb:mode={rank_mode}:energy={energy_ratio}:eps={eps}"
        compute_device = self._get_merge_compute_device()
        compute_dtype = self._get_pgsb_compute_dtype()
        if cache_key in self.cache:
            core_stack_cpu, Q_left_cpu, Q_right_cpu, meta = self.cache[cache_key]
            return (
                core_stack_cpu.to(device=compute_device, dtype=compute_dtype),
                Q_left_cpu.to(device=compute_device, dtype=compute_dtype),
                Q_right_cpu.to(device=compute_device, dtype=compute_dtype),
                meta,
            )

        A_list, B_list = zip(*[ftm[key] for ftm in ftms_params_ab])
        resolved_key = self._resolve_pretrained_key(key, pretrained_sd)
        W0 = self._to_merge_compute(pretrained_sd[resolved_key], compute_device, compute_dtype)
        m, n = W0.shape

        left_factors, singular_values, right_factors = self.compact_lora_svd_batch(
            A_list,
            B_list,
            eps=eps,
            compute_device=compute_device,
            compute_dtype=compute_dtype,
        )
        block_spec = self.infer_attention_block_spec(key, W0.shape)
        Q_left_lora, left_meta = self.build_pgsb_consensus_basis(
            left_factors,
            m,
            block_spec['left_ranges'],
            prior_matrix=None,
            side='left',
            merge_config=merge_config,
        )
        Q_right_lora, right_meta = self.build_pgsb_consensus_basis(
            right_factors,
            n,
            block_spec['right_ranges'],
            prior_matrix=None,
            side='right',
            merge_config=merge_config,
        )

        reduced_core_list = []
        for U_t, S_t, V_t in zip(left_factors, singular_values, right_factors):
            if S_t.numel() == 0:
                core = torch.zeros(
                    (Q_left_lora.shape[1], Q_right_lora.shape[1]),
                    dtype=compute_dtype,
                    device=compute_device,
                )
            else:
                left_proj = Q_left_lora.T @ U_t
                right_proj = V_t.T @ Q_right_lora
                core = (left_proj * S_t.unsqueeze(0)) @ right_proj
            reduced_core_list.append(core)

        transport_matrix = Q_left_lora.T @ W0 @ Q_right_lora
        attention_priors = self._build_llama_attention_coupled_priors(
            key=key,
            pretrained_sd=pretrained_sd,
            W0=W0,
            Q_left_lora=Q_left_lora,
            Q_right_lora=Q_right_lora,
            compute_device=compute_device,
            compute_dtype=compute_dtype,
            eps=eps,
        )
        canonical_core_list, O_left, O_right, gauge_meta = self.build_pgsb_transport_intersection_gauge(
            reduced_core_list,
            transport_matrix,
            left_meta,
            right_meta,
            block_spec,
            attention_priors=attention_priors,
            eps=eps,
        )
        Q_left = Q_left_lora @ O_left if O_left.numel() > 0 else Q_left_lora[:, :0]
        Q_right = Q_right_lora @ O_right if O_right.numel() > 0 else Q_right_lora[:, :0]

        meta = {
            'block_spec': block_spec,
            'left': left_meta,
            'right': right_meta,
            'task_ranks': [int(s.numel()) for s in singular_values],
            'gauge': gauge_meta,
        }
        core_stack = self._pgsb_stack_tensors(canonical_core_list)
        cached_value = (core_stack.detach().cpu(), Q_left.detach().cpu(), Q_right.detach().cpu(), meta)
        self.cache[cache_key] = cached_value
        return core_stack, Q_left, Q_right, meta


    def _merge_tensors(self, tensors, merge_config, cache_key=None):
        merge_method = merge_config.get('merge_method')
        merge_cache_key = self._get_merge_result_cache_key(cache_key, merge_config)
        target_device = tensors.device if isinstance(tensors, torch.Tensor) else tensors[0].device
        target_dtype = tensors.dtype if isinstance(tensors, torch.Tensor) else tensors[0].dtype
        if merge_cache_key is not None and merge_cache_key in self.cache:
            cached_tensor = self.cache[merge_cache_key]
            return cached_tensor.to(device=target_device, dtype=target_dtype)

        if merge_method == 'tv':
            result = tensors.sum(dim=0) if isinstance(tensors, torch.Tensor) else torch.stack(tensors).sum(dim=0)
        elif merge_method == 'ties':
            result = self._process_ties(tensors, merge_config.get('topK', 10))
        elif merge_method == 'dare-ties':
            tensors_dare = self.get_dare_delta_w(tensors, merge_config.get('dare_pruning_coeffs', 0.3), merge=False)
            result = self._process_ties(tensors_dare, merge_config.get('topK', 10))
        elif merge_method == 'tsv':
            result = self.get_tsv_delta_w(tensors, merge_config=merge_config, cache_key=cache_key)
        elif merge_method == 'cart':
            result = self.get_cart_delta_w(
                tensors,
                merge_config.get('cart_pruning_rank', 0.04),
                merge_config.get('cart_scaling_coeffs', 0.1)
            )
        else:
            raise ValueError(f"Unknown merge_method: {merge_method}")

        if (
            hasattr(result, 'detach')
            and merge_cache_key is not None
            and int(getattr(result, 'numel', lambda: 0)()) <= 2_000_000
        ):
            self.cache[merge_cache_key] = result.detach().cpu()
        return result.to(device=target_device, dtype=target_dtype) if hasattr(result, 'to') else result


    def merge(self, merge_config):
        merge_config = canonicalize_merge_config(merge_config)
        print(f"Merging using {merge_config.get('merge_space')} - {merge_config.get('merge_method')}")
        print(f"Isotropizing = {merge_config.get('isotropize', False)}")
        log_merge_ranks = merge_config.get('log_merge_ranks', False)

        # Determine merge space and prepare parameters
        if merge_config.get('merge_space') in ('core', 'pgsb'):
            ftms_params_ab = [ftm.get_ft_ab_parameters() for ftm in self.ftms_params]
            relevant_ab_keys = self.ftms_params[0].get_ft_ab_parameters().keys()
        else:
            ftms_task_dirs = [ftm.get_ft_parameters() for ftm in self.ftms_params]

        # with newer peft versions, the keys may not have '.base_layer' in them
        # so we handle it by replacing it with an empty string (a cleaner solution would be nice)
        all_keys = self.pretrained_model.state_dict().keys()
        new_sd = deepcopy(self.pretrained_model)
        param_dict = dict(new_sd.named_parameters())

        avg_ranks = [] if log_merge_ranks else None
        if merge_config.get('merge_space') == 'full':
            compute_device = self._get_merge_compute_device()
            print(f"Full merge compute device: {compute_device}")
            for key in tqdm(all_keys, desc="Merging full space"):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_task_dirs[0]:
                    tensor_list = self._prepare_tensors_for_merge(
                        [ft_dir[key_base] for ft_dir in ftms_task_dirs],
                        compute_device,
                        merge_config.get('merge_method'),
                    )
                    delta_w = self._merge_tensors(tensor_list, merge_config, cache_key=f"full:{key_base}")
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                    if avg_ranks is not None:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                    self._apply_delta(param_dict, key, delta_w.cpu())
        
        elif merge_config.get('merge_space') == 'knots':
            compute_device = self._get_merge_compute_device()
            print(f"KnOTS merge compute device: {compute_device}")
            for key in tqdm(all_keys, desc="Merging knots space"):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_task_dirs[0]:
                    knots_cache_key = f"knots:{key_base}:components"
                    knots_cache_path = self._get_disk_cache_path(merge_config, knots_cache_key)
                    if knots_cache_key in self.cache or (knots_cache_path is not None and knots_cache_path.exists()):
                        tensor_list = None
                    else:
                        tensor_list = [deepcopy(ft_dir[key_base]) for ft_dir in ftms_task_dirs]
                    U, S, Vs = self.get_knots_components(
                        tensor_list,
                        merge_config=merge_config,
                        cache_key=knots_cache_key,
                    )
                    U = U.to(compute_device)
                    S = S.to(compute_device)
                    Vs = self._prepare_tensors_for_merge(Vs, compute_device, merge_config.get('merge_method'))

                    Vs_merged = self._merge_tensors(Vs, merge_config, cache_key=f"knots:{key_base}:merge")

                    delta_w = U @ torch.diag(S) @ Vs_merged
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                    if avg_ranks is not None:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                    self._apply_delta(param_dict, key, delta_w.cpu())

        elif merge_config.get('merge_space') == 'core':
            compute_device = self._get_merge_compute_device()
            print(f"Core merge compute device: {compute_device}")
            for key in tqdm(all_keys, desc="Merging core space"):
                key_base = key.replace('.base_layer', '')
                if key_base in relevant_ab_keys:
                    M_list, U_B_ref, Vh_A_ref = self.get_core_matrices(ftms_params_ab, key_base, merge_config)

                    M_list = self._prepare_tensors_for_merge(M_list, compute_device, merge_config.get('merge_method'))
                    U_B_ref = U_B_ref.to(compute_device)
                    Vh_A_ref = Vh_A_ref.to(compute_device)
                    M_merged = self._merge_tensors(M_list, merge_config, cache_key=f"core:{key_base}")

                    if merge_config.get('isotropize', False):
                        M_merged = isotropize_matrix(M_merged)
                    M_merged = M_merged.to(device=compute_device, dtype=U_B_ref.dtype)

                    delta_W = U_B_ref @ M_merged @ Vh_A_ref

                    if avg_ranks is not None:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_W).item())
                    self._apply_delta(param_dict, key, delta_W.cpu())

        elif merge_config.get('merge_space') == 'pgsb':
            pretrained_sd = self.pretrained_model.state_dict()
            compute_device = self._get_merge_compute_device()
            print(f"PGSB merge compute device: {compute_device}")
            if compute_device.type == 'cuda':
                try:
                    torch.set_float32_matmul_precision('high')
                except AttributeError:
                    pass
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.allow_tf32 = True
            pgsb_left_ranks = []
            pgsb_right_ranks = []
            for key in tqdm(all_keys, desc="Merging PGSB"):
                key_base = key.replace('.base_layer', '')
                if key_base in relevant_ab_keys:
                    M_list, P_left, P_right, pgsb_meta = self.get_pgsb_components(
                        ftms_params_ab,
                        key_base,
                        merge_config,
                        pretrained_sd,
                    )

                    M_list = self._prepare_tensors_for_merge(M_list, compute_device, merge_config.get('merge_method'))
                    P_left = P_left.to(compute_device)
                    P_right = P_right.to(compute_device)
                    M_merged = self._merge_tensors(M_list, merge_config, cache_key=f"pgsb:{key_base}")
                    expected_shape = (P_left.shape[1], P_right.shape[1])
                    if tuple(M_merged.shape) != expected_shape:
                        M_merged = self._merge_tensors(M_list, merge_config, cache_key=None)
                    if tuple(M_merged.shape) != expected_shape:
                        raise RuntimeError(
                            f"PGSB merged core shape mismatch for {key_base}: "
                            f"expected {expected_shape}, got {tuple(M_merged.shape)}"
                        )

                    if merge_config.get('isotropize', False):
                        M_merged = isotropize_matrix(M_merged)
                    M_merged = M_merged.to(device=compute_device, dtype=P_left.dtype)

                    delta_W = P_left @ M_merged @ P_right.T
                    if avg_ranks is not None:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_W).item())
                    pgsb_left_ranks.append(int(pgsb_meta['left']['rank']))
                    pgsb_right_ranks.append(int(pgsb_meta['right']['rank']))
                    self._apply_delta(param_dict, key, delta_W.cpu())

            if pgsb_left_ranks:
                print(
                    f"PGSB left rank: mean={sum(pgsb_left_ranks)/len(pgsb_left_ranks):.1f}, "
                    f"min={min(pgsb_left_ranks)}, max={max(pgsb_left_ranks)}"
                )
            if pgsb_right_ranks:
                print(
                    f"PGSB right rank: mean={sum(pgsb_right_ranks)/len(pgsb_right_ranks):.1f}, "
                    f"min={min(pgsb_right_ranks)}, max={max(pgsb_right_ranks)}"
                )

        else:
            raise ValueError(f"Unknown merge_space: {merge_config.get('merge_space')}")

        if avg_ranks:
            print(f"Average rank of delta_W across all layers: {sum(avg_ranks)/len(avg_ranks):.2f}")

        gc.collect()
        return new_sd



def isotropize_matrix(matrix):
    """Iso-C: replace all singular values by their arithmetic mean."""
    matrix64 = matrix.to(torch.float64)
    U, S, Vh = torch.linalg.svd(matrix64, full_matrices=False)
    if S.numel() == 0:
        return matrix64
    S_iso = S.mean() * torch.ones_like(S)
    return U @ torch.diag(S_iso) @ Vh


def get_merge_handler(rep_type):
    """Return the repository's layerwise matrix merger."""
    if rep_type != 'matrix_per_layer':
        raise ValueError('Only representation="matrix_per_layer" is supported.')
    return MatrixPerLayerMerger
