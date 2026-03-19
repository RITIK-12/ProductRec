# LoViF — Multimodal Product Rating Prediction

A lightweight multimodal fusion model that predicts product quality ratings from images and text metadata. Built for the **CVPR 2025 Efficient VLM Workshop** challenge, by Ritik Bompilwar & Sahil Faisal.

## Architecture

The pipeline follows a dual-encoder → fusion → regression design:

1. **Encoder**: A frozen [SigLIP2 ViT-B/16](https://huggingface.co/timm/ViT-B-16-SigLIP2) backbone encodes both images and tokenised product text (category, title, description, features) into 768-d embeddings. The last 2 vision Transformer blocks are unfrozen for task-specific fine-tuning while the text encoder stays fully frozen — this keeps the trainable parameter count low.

2. **Fusion**: Image and text embeddings are projected into a shared 512-d space with learned type embeddings, then concatenated with 4 learnable query tokens. This sequence passes through 2 Transformer blocks with self-attention, enabling deep cross-modal interaction between visual and textual signals.

3. **Pooling**: A gated mechanism learns a soft routing between image and text representations. Query token outputs are averaged and added as an auxiliary signal.

4. **Output Head**: The gated features plus a raw cosine similarity scalar are fed through a 3-layer MLP (512→256→64→1) with GELU activations and dropout, followed by sigmoid scaling to bound predictions in [1, 5].

5. **Loss**: A composite objective combining MSE (per-sample accuracy), 1-PLCC (distribution-level correlation), and margin ranking loss (pairwise ordering). This directly optimises for the competition metric while maintaining stable gradients.

**Model stats**: ~383M total params (7.5M trainable) · 20.7 GFLOPs

## Model

Trained model weights can be downloaded from [ritik12/ProductRec](https://huggingface.co/ritik12/ProductRec).

## Project Structure

```
lovif/
├── config.py              # All hyperparameters, paths, and constants
├── train.py               # Training entrypoint
├── evaluate.py            # Test inference and submission generation
├── requirements.txt
├── data/
│   ├── download.py        # Parallel image downloader + test set fetcher
│   ├── dataset.py         # PyTorch Datasets and collate functions
│   └── preprocess.py      # Text cleaning and tokenisation helpers
├── model/
│   ├── fusion.py          # SigLIP2FusionModel + FusionBlock
│   └── loss.py            # FusionRatingLoss (MSE + PLCC + ranking)
└── engine/
    ├── train.py           # Training and evaluation loops
    └── inference.py       # Test prediction and submission CSV generation
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU (tested on L4 / A100). Training data and test set are downloaded automatically on first run.

## Training

```bash
python train.py
```

This will:
1. Download the [Amazon Products 2023](https://huggingface.co/datasets/milistu/AMAZON-Products-2023) dataset from HuggingFace
2. Download product images in parallel (64 workers, cached to `./cache/images/`)
3. Perform a stratified 90/10 train/val split
4. Train for up to 20 epochs with early stopping (patience=5)
5. Save the best checkpoint to `./output/best_model.pt`

Training logs are sent to [Weights & Biases](https://wandb.ai). To disable:

```bash
python train.py --no-wandb
```

**Key training details**:
- Warmup: 2 epochs of linear warmup
- LR: 5e-6 for unfrozen encoder layers, 3e-4 for fusion+head
- Scheduler: ReduceLROnPlateau on validation PLCC+ (factor=0.5, patience=2)
- Mixed precision (FP16) enabled by default
- Gradient clipping at 1.0

## Evaluation

```bash
python evaluate.py --checkpoint output/best_model.pt
```

This will:
1. Download the test set (setB) from the [competition HuggingFace repo](https://huggingface.co/datasets/Kirin0010/CVPR_workshop_efficiencyVLM)
2. Load the checkpoint and run inference over all test images
3. Aggregate multi-image products by averaging scores
4. Write `submission.csv` to `./output/`

To include FLOPs analysis in the output (requires `fvcore`):

```bash
python evaluate.py --checkpoint output/best_model.pt --compute-flops
```

Custom output path:

```bash
python evaluate.py --checkpoint output/best_model.pt --output my_submission.csv
```

## Configuration

All hyperparameters live in `config.py`. Key ones to tune:

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 512 | Reduce if OOM on smaller GPUs |
| `head_lr` | 3e-4 | Learning rate for fusion + head |
| `unfreeze_vision_layers` | 2 | Number of vision blocks to fine-tune |
| `epochs` | 20 | Max epochs (early stopping usually triggers ~12) |
| `plcc_weight` | 2.0 | Weight for PLCC loss term |
| `dropout` | 0.2 | Dropout in fusion blocks and head |

Paths can be overridden via environment variables:
```bash
export LOVIF_IMG_CACHE=/data/img_cache
export LOVIF_SAVE_DIR=/data/output
export LOVIF_TEST_ROOT=/data/test
```

## Submission Format

The output CSV has the columns expected by the challenge leaderboard:

| Column | Description |
|---|---|
| `item_id` | Product ASIN |
| `score` | Predicted rating [1.0, 5.0] |
| `params` | Total model parameters (M) |
| `flops` | Forward pass FLOPs (G) |

## Acknowledgements

1. **Training dataset**: We use the Amazon Products dataset from [Hou et al. (2024)](https://arxiv.org/abs/2403.03952) — *Bridging Language and Items for Retrieval and Recommendation*.

2. **Validation/test dataset**: The validation and test splits are from the [CVPR Workshop Efficiency VLM dataset](https://huggingface.co/datasets/Kirin0010/CVPR_workshop_efficiencyVLM). Please use all data responsibly in accordance with the dataset license, terms of use, and applicable ethical guidelines.

3. **SigLIP**: Our vision encoder builds on [SigLIP](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023) — *Sigmoid Loss for Language Image Pre-Training*.

### References

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}

@article{zhai2023siglip,
  title={Sigmoid Loss for Language Image Pre-Training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  journal={arXiv preprint arXiv:2303.15343},
  year={2023}
}
```
