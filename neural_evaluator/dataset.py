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
        item = {"boards": torch.tensor(self.boards[idx], dtype=torch.float32),
                "policies": torch.tensor(self.policies[idx], dtype=torch.float32),
                "success": torch.tensor(self.success[idx], dtype=torch.float32)}

        return item

    def __len__(self):
        return self.n_samples
