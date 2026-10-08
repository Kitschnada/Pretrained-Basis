"""Reduction utility preserving the main-experiment summation order."""

import torch


def chunked_sum(tensor, chunk_size=10000):
    num_chunks = tensor.size(0) // chunk_size + (1 if tensor.size(0) % chunk_size != 0 else 0)
    total_sum = torch.zeros_like(tensor[0])

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tensor.size(0))
        chunk = tensor[start_idx:end_idx]

        # Add the sum of the current chunk to the total sum
        total_sum += torch.sum(chunk, dim=0)

    return total_sum
