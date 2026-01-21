import torch
import os
import glob
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .loss_function import compute_forge_loss


class VGTTrainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.batch_size = config["batch_size"]
        self.total_steps = config["total_steps"]
        self.lr = config["lr"]
        self.vocab_size = config["vocab_size"]
        self.checkpoint_dir = config.get("checkpoint_dir", ".")
        self.log_interval = config.get("log_interval", 100)
        self.save_interval = config.get("save_interval", 5000)

        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=config.get("weight_decay", 0.01)
        )
        self.scaler = torch.amp.GradScaler('cuda')
        self.controller = config["forge_controller"]

        self.start_step = 0
        self._load_checkpoint()

    def _load_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "vgt_8L_step_*.pth"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            self.start_step = int(latest.split("_")[-1].split(".")[0])
            self.model.load_state_dict(torch.load(latest, map_location=self.device))
            print(f"♻️ 8L 续传成功: {latest}")

    def train(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size)
        print("🚀 VGT-8L-768D 启动 | 稳定 Forge 模式")

        for step, (x, y) in enumerate(loader):
            curr_step = step + self.start_step
            if curr_step > self.total_steps:
                break

            x, y = x.to(self.device), y.to(self.device)

            with torch.amp.autocast("cuda"):
                logits, h_states = self.model(x)
                alpha, mode = self.controller.get_alpha(curr_step)

                total_loss, loss_components = compute_forge_loss(
                    logits=logits,
                    targets=y,
                    h_states=h_states,
                    embedding_layer=self.model.embedding,
                    vocab_size=self.vocab_size,
                    alpha=alpha
                )

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if curr_step % self.log_interval == 0:
                ce_loss = loss_components["ce_loss"]
                cos_sim = loss_components["cos_sim"]
                h_norm = loss_components["h_norm"]
                alpha_val = loss_components["alpha"]
                print(
                    f"[{curr_step:6d}] {mode} | "
                    f"CE: {ce_loss:.4f} | "
                    f"Cos: {cos_sim:.3f} | "
                    f"H-Norm: {h_norm:.3f} | "
                    f"α: {alpha_val:.1f}"
                )

            if curr_step > 0 and curr_step % self.save_interval == 0:
                save_path = os.path.join(self.checkpoint_dir, f"vgt_8L_step_{curr_step}.pth")
                torch.save(self.model.state_dict(), save_path)