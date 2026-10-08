# Model interfaces

`huggingface_clip.py` provides the ViT-B/32 and ViT-L/14 CLIP vision towers.
`huggingface_siglip.py` provides SigLIP, including RGB conversion for grayscale
MNIST inputs. Both preserve the checkpoint parameter names, projection heads,
and preprocessing used by the main experiments.

The Llama-3-8B NLI sequence-classification model is prepared in `utils.py`.
Its six task heads are loaded separately by the NLI evaluators.
