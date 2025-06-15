# pylint: disable=not-callable, no-member, no-name-in-module

import torch


class Connect4Dataset(torch.utils.data.Dataset):
    def __init__(self, boards, policies, success, training=False):
        self.boards = boards
        self.policies = policies
        self.success = success
        self.training = training
        self.n_samples = len(self.success)

    def __getitem__(self, idx):
        # Add transforms for data augmentation
        idx = idx % self.n_samples
        tmp_boards = torch.tensor(self.boards[idx])
        # if self.training and random.random() < 0.5:
        #     tmp_boards = torch.flip(tmp_boards, [1])
        #     # I think we need to flip the policies as well
        boards = torch.zeros(2, *tmp_boards.shape, dtype=torch.float32)
        boards[0, tmp_boards == 1] = 1
        boards[1, tmp_boards == 2] = 1
        item = {
            "boards": boards,
            "policies": torch.tensor(self.policies[idx], dtype=torch.float32),
            "success": torch.tensor(self.success[idx], dtype=torch.float32),
        }

        return item

    def __len__(self):
        return self.n_samples
