import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from neural_scripts.custom_loss import AlphaLoss
from neural_scripts.dataset import Connect4Dataset


@dataclass
class TrainingArgs:
    train_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 50
    learning_rate: float = 0.0005
    print_progress: bool = False
    model_output_path: Optional[str] = None
    from_pretrained: Optional[str] = None


class Trainer:
    def __init__(
        self,
        model,
        train_dataset: Connect4Dataset,
        test_dataset: Connect4Dataset,
        training_args: TrainingArgs = TrainingArgs(),
    ):
        self.model = model.to(training_args.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.training_args = training_args

        if self.training_args.from_pretrained and os.path.isfile(self.training_args.from_pretrained):
            print(f"Loading from pretrained model {self.training_args.from_pretrained}")
            self.model.load_state_dict(torch.load(self.training_args.from_pretrained))

        # --- switch to AdamW ---
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=1e-4,  # you can tune this
        )

        # we'll set scheduler in `train()` once we know total steps
        self.scheduler = None
        self.loss_function = AlphaLoss(weight=1)

    def train(self):
        # build DataLoader
        data_loader = DataLoader(self.train_dataset, batch_size=self.training_args.batch_size, shuffle=True)
        total_steps = len(data_loader) * self.training_args.train_epochs

        # --- linear warmup for 10% of total steps, then cosine decay ---
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # cosine anneal from 1 down to 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # initial eval
        self.infer(epoch=-1)
        self.model.train()

        for epoch in range(self.training_args.train_epochs):
            total_loss = 0.0
            for step, batch in enumerate(data_loader):
                boards = batch["boards"].to(self.training_args.device)
                policies = batch["policies"].to(self.training_args.device)
                wins = batch["success"].to(self.training_args.device)

                self.optimizer.zero_grad()
                output_pol, output_val = self.model(boards)
                loss = self.loss_function(output_pol, output_val, policies, wins)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()  # <— update LR each batch

                total_loss += loss.item()

            avg_train_loss = total_loss / len(data_loader)
            if self.training_args.print_progress:
                print(f"Epoch {epoch}  Train loss: {avg_train_loss:.4f}  LR: {self.scheduler.get_last_lr()[0]:.2e}")
            self.infer(epoch=epoch)

        if self.training_args.model_output_path:
            torch.save(self.model.state_dict(), self.training_args.model_output_path)

    def infer(self, test_dataset=None, epoch=None):
        self.model.eval()
        data_loader = DataLoader(
            test_dataset if test_dataset else self.test_dataset, batch_size=self.training_args.batch_size, shuffle=False
        )

        total_loss = 0
        for batch in data_loader:
            boards = batch["boards"].to(self.training_args.device)
            policies = batch["policies"].to(self.training_args.device)
            wins = batch["success"].to(self.training_args.device)

            with torch.no_grad():
                output_pol, output_pos = self.model(boards)
                total_loss += self.loss_function(output_pol, output_pos, policies, wins).item()

        total_loss = total_loss / len(data_loader)
        self.model.train()

        if self.training_args.print_progress:
            print(f"\n Loss/test: {total_loss} - Epoch {epoch}")
