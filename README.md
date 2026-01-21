# Forge-VGT: Vision-Guided Transformer with Forging Optimization

A modular, open-source implementation of the VGT-8L model with dynamic training stabilization via the Forge Controller.

## 📦 Features

- **Modular Architecture**: Clean separation of model, dataset, training loop, and optimization logic.
- **Forge Training Strategy**: Adaptive loss scaling (`α`) with warm-up, compaction, and annealing phases.
- **Stable Residual Blocks**: GRU-based residual blocks with LayerNorm and dropout.
- **Streaming Dataset**: Efficient line-by-line JSONL data loading for large corpora.
- **Mixed-Precision Training**: AMP (Automatic Mixed Precision) support for faster GPU training.
- **Checkpointing**: Automatic resume from latest checkpoint.
- **Pluggable Loss Function**: Core `compute_forge_loss` is decoupled for easy modification or replacement.

## 🛠️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA-compatible GPU (for training)

Install dependencies:
```bash
pip install torch
```

## 📁 Project Structure

```
Forge-VGT/
├── models/               # Model definitions
│   ├── __init__.py
│   └── vgt_8l.py
├── data/                 # Dataset utilities
│   ├── __init__.py
│   └── stream_dataset.py
├── training/             # Training logic
│   ├── __init__.py
│   ├── forge_controller.py
│   ├── loss_function.py  # ← Pluggable loss core
│   └── trainer.py
├── config/               # Configuration files
│   └── train_config.json
├── vocab.json            # Vocabulary file (required)
├── train_encyclopedia.json  # Training data (JSONL format)
├── train.py              # Main entry point
└── README.md
```

## 🔌 Core Loss Function (Pluggable Design)

The heart of Forge-VGT is its composite loss, now fully modular in [`training/loss_function.py`](training/loss_function.py):

```python
def compute_forge_loss(logits, targets, h_states, embedding_layer, vocab_size, alpha):
    # 1. Standard cross-entropy
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    # 2. Hidden state norm regularization
    h_norm = torch.sqrt(torch.mean(h_states ** 2))

    # 3. Cosine alignment with target embeddings
    with torch.no_grad():
        target_emb = embedding_layer(targets)
    cos_sim = F.cosine_similarity(h_states, target_emb, dim=-1).mean()
    cos_loss = 1.0 - cos_sim

    # Combine with dynamic Forge scaling
    total_loss = ce_loss + alpha * 0.15 * h_norm + alpha * 0.40 * cos_loss
    return total_loss, { ... }
```

This design allows researchers to:
- Swap in alternative regularization terms
- Adjust weighting coefficients without touching the trainer
- Reuse the loss in other architectures

## ▶️ Quick Start

1. Prepare your vocabulary (`vocab.json`) and training data (`train_encyclopedia.json` in JSONL format).
2. Run training:
   ```bash
   python train.py
   ```
3. Checkpoints will be saved as `vgt_8L_step_{step}.pth`.

## 📜 License

MIT License. See `LICENSE` for details.

## 🌟 Acknowledgements

Inspired by adaptive regularization and representation forging techniques in modern language modeling.