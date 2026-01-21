import json
import os
import torch
from models import VGT_8L_Engine
from data import StreamDataset
from training import VGTForgeController, VGTTrainer


def main():
    # Load configuration
    with open("config/train_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load vocabulary
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    config["vocab_size"] = len(vocab)

    # Initialize model
    model = VGT_8L_Engine(
        vocab_size=config["vocab_size"],
        d_model=768,
        num_layers=8
    )

    # Initialize dataset
    dataset = StreamDataset(
        file_path="train_encyclopedia.json",
        vocab=vocab,
        seq_len=256
    )

    # Initialize forge controller
    forge_controller = VGTForgeController(
        total_steps=config["total_steps"],
        warmup=config["warmup_steps"]
    )
    config["forge_controller"] = forge_controller

    # Initialize and run trainer
    trainer = VGTTrainer(model, dataset, config)
    trainer.train()


if __name__ == "__main__":
    main()