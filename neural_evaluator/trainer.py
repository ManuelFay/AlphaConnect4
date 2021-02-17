# pylint: disable=not-callable, no-member, no-name-in-module
from dataclasses import dataclass
from typing import Optional
import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.adam import Adam

from neural_evaluator.dataset import Connect4Dataset
from neural_evaluator.custom_loss import AlphaLoss


@dataclass
class TrainingArgs:
    train_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 50
    learning_rate: float = 0.005
    print_progress: bool = False
    model_output_path: Optional[str] = None
    from_pretrained: Optional[str] = None


class Trainer:
    def __init__(self,
                 model,
                 train_dataset: Connect4Dataset,
                 test_dataset: Connect4Dataset,
                 training_args: TrainingArgs = TrainingArgs()):

        self.model = model.to(training_args.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.training_args = training_args

        if self.training_args.from_pretrained and os.path.isfile(self.training_args.from_pretrained):
            print(f"Loading from pretrained model {self.training_args.from_pretrained}")
            self.model.load_state_dict(torch.load(self.training_args.from_pretrained))

        self.optimizer = Adam(self.model.parameters(), lr=self.training_args.learning_rate)
        self.loss_function = AlphaLoss(weight=1)
        self.writer = SummaryWriter()

    def train(self):
        self.infer(epoch=-1)
        self.model.train()
        data_loader = DataLoader(self.train_dataset, batch_size=self.training_args.batch_size, shuffle=True)

        for epoch in tqdm(range(self.training_args.train_epochs)):
            total_loss = 0
            for batch in data_loader:

                boards = batch['boards'].to(self.training_args.device)
                policies = batch['policies'].to(self.training_args.device)
                wins = batch['success'].to(self.training_args.device)

                self.model.zero_grad()
                output_pol, output_pos = self.model(boards)
                loss = self.loss_function(output_pol, output_pos, policies, wins)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            total_loss = total_loss/len(data_loader)
            self.writer.add_scalar("Loss/train", total_loss, epoch)
            if self.training_args.print_progress:
                print(f"\n Loss/train: {total_loss} - {epoch}")
            self.infer(epoch=epoch)

        if self.training_args.model_output_path:
            torch.save(self.model.state_dict(), self.training_args.model_output_path)

    def infer(self, test_dataset=None, epoch=None):
        self.model.eval()
        data_loader = DataLoader(test_dataset if test_dataset else self.test_dataset,
                                 batch_size=self.training_args.batch_size,
                                 shuffle=False)

        total_loss = 0
        for batch in data_loader:
            boards = batch['boards'].to(self.training_args.device)
            policies = batch['policies'].to(self.training_args.device)
            wins = batch['success'].to(self.training_args.device)

            with torch.no_grad():
                output_pol, output_pos = self.model(boards)
                total_loss += self.loss_function(output_pol, output_pos, policies, wins).item()

        total_loss = total_loss / len(data_loader)
        self.model.train()

        if epoch is not None:
            self.writer.add_scalar("Loss/test", total_loss, epoch)
        if self.training_args.print_progress:
            print(f"\n Loss/test: {total_loss} - {epoch}")
