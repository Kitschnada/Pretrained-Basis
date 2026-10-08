import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .nli_downloads import TASK_IDS, TASK_KEYS, load_nli_dataset

ROOT = os.getenv("PGSB_NLI_DATA_CACHE", "")  # Preserve the existing cache unless explicitly overridden.

task_to_keys = TASK_KEYS
task_ids = TASK_IDS


class HuggingFaceDataset:
    def __init__(self,
                 location=None,
                 task=None,
                 model_name_or_path=None,
                 batch_size=32,
                 num_workers=16):

        self.batch_size = batch_size
        self.num_workers = num_workers
        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dataset = load_nli_dataset(task, cache_dir=ROOT, location=location)
        sentence1_key, sentence2_key = task_to_keys[task]

        def tokenize_function(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            outputs = self.tokenizer(*args, truncation=True, max_length=2000)
            return outputs

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[t for t in dataset['train'].column_names if t != "label"]
        )

        self.tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        self.tokenized_datasets = self.tokenized_datasets.filter(lambda example: example["labels"] == 0 or example["labels"] == 1 or example["labels"] == 2)
        for split in ("train", "validation", "test"):
            if task in ("mnli", "qnli", "rte") and split == "test":
                continue  # Unlabeled GLUE challenge splits are not evaluated.
            if not len(self.tokenized_datasets[split]):
                raise ValueError("{} {}: no labeled examples after filtering".format(task, split))

        if "train" in self.tokenized_datasets:
            self.train_loader = DataLoader(self.tokenized_datasets["train"], shuffle=True, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)
        if "test" in self.tokenized_datasets:
            self.test_loader = DataLoader(self.tokenized_datasets["test"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)
        if "validation" in self.tokenized_datasets:
            self.val_loader = DataLoader(self.tokenized_datasets["validation"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)

    def collate_fn(self, examples):
        return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")
