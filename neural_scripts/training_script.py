import numpy as np

from alphaconnect4.constants.constants import COLUMN_COUNT, ROW_COUNT
from alphaconnect4.interfaces.naive_nn import NaiveNet
from alphaconnect4.interfaces.transformer_nn import ConvTransformerNet
from neural_scripts.dataset import Connect4Dataset
from neural_scripts.trainer import Trainer, TrainingArgs

data_train = np.load("./data/training_1a.npy", allow_pickle=True)
data_test = np.load("./data/training_1b.npy", allow_pickle=True)
print(f"Number of samples: {data_train.shape[1]} / {data_test.shape[1]}")

train_set = Connect4Dataset(data_train[0], data_train[1], data_train[2], training=True)
test_set = Connect4Dataset(data_test[0], data_test[1], data_test[2], training=False)


args = TrainingArgs(
    train_epochs=10,
    batch_size=50,
    print_progress=True,
    # from_pretrained="./models/model_0.pth",
    model_output_path="./models/model_naive1.pth",
)
trainer = Trainer(
    model=ConvTransformerNet(ROW_COUNT, COLUMN_COUNT) if False else NaiveNet(ROW_COUNT, COLUMN_COUNT),
    train_dataset=train_set,
    test_dataset=test_set,
    training_args=args,
)

trainer.train()
