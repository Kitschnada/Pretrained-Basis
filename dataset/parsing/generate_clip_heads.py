"""Generate the existing CLIP class-vector heads for the eight vision tasks."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from dataset.templates import get_templates
from utils import get_config_from_name, prepare_data


def build_classification_head(model, tokenizer, classnames, template, device):
    """Keep the original prompt averaging, normalization and logit scale."""
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Building classification head"):
            embeddings = []
            for prompt in template:
                tokenized = tokenizer(prompt(classname))
                tokenized = {
                    key: torch.tensor(value).to(device).reshape(1, -1)
                    for key, value in tokenized.items()
                }
                embeddings.append(model.text_projection(model.text_model(**tokenized)[1]))
            embeddings = torch.concat(embeddings, dim=0)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= model.logit_scale.exp()
        return torch.transpose(zeroshot_weights.squeeze().float(), 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="vit_b32.py", choices=("vit_b32.py", "vit_l14.py"))
    parser.add_argument("--output-dir", help="Default: the model's configured class-head directory.")
    parser.add_argument("--cache-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--force", action="store_true", help="Explicitly overwrite existing heads.")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.data_root:
        os.environ["PGSB_VISION_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config_from_name(args.config, device=device)
    pending = []
    for entry in config["dataset"]:
        path = (
            Path(args.output_dir) / (entry["name"] + "_head.pt")
            if args.output_dir else Path(entry["clip_encodings"])
        )
        if path.exists() and not args.force:
            print(f"Keeping existing head: {path}")
        else:
            pending.append((entry, path))
    if not pending:
        return
    model_name = config["model"]["base_type"]
    cache_dir = args.cache_dir or config["model"]["cachedir"]
    model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    for entry, path in pending:
        entry = dict(entry)
        entry["train_preprocess"] = processor.image_processor
        entry["eval_preprocess"] = processor.image_processor
        loaders = prepare_data(entry, device=device)
        head = build_classification_head(
            model, processor.tokenizer, loaders["test"]["class_names"],
            get_templates(entry["name"]), device,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head.cpu(), path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
