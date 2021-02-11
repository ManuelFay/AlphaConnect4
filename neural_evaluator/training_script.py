import numpy as np

from neural_evaluator.dataset import Connect4Dataset
from neural_evaluator.stub_nn import StubNet
from neural_evaluator.naive_nn import NaiveNet
from neural_evaluator.trainer import Trainer, TrainingArgs

from gameplay.constants import ROW_COUNT, COLUMN_COUNT


def detect_full_cols(board):
    board = ((board != 0).sum(axis=0) == 6).squeeze()
    return np.where(board)[0].tolist()


def normalize_policies(boards, policies):
    new_pols = []
    for board, pol in zip(boards, policies):
        pol2 = pol.copy()
        for idx in detect_full_cols(board):
            pol2.insert(idx, 0.)
        new_pols.append(pol2)

    return np.array(new_pols)


data = np.load("../training_0.npy", allow_pickle=True)
print(f"Number of training samples: {data.shape[1]}")

new_policies = normalize_policies(data[0], data[1])

train_set = Connect4Dataset(data[0], new_policies, data[2], training=True)
test_set = Connect4Dataset(data[0], new_policies, data[2], training=False)


args = TrainingArgs(
    train_epochs=10,
    batch_size=500,
    print_progress=True,
    from_pretrained=None,
    model_output_path="/home/manu/perso/RL_Connect4/model_0b.pth"
)
trainer = Trainer(model=NaiveNet(ROW_COUNT, COLUMN_COUNT),
                  train_dataset=train_set,
                  test_dataset=test_set,
                  training_args=args
                  )

trainer.train()
