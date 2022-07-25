import numpy as np

from neural_scripts.dataset import Connect4Dataset
from alphaconnect4.interfaces.naive_nn import NaiveNet
from neural_scripts.trainer import Trainer, TrainingArgs

from alphaconnect4.interfaces.connect4.constants import ROW_COUNT, COLUMN_COUNT


data_train = np.load("../data/training_2c.npy", allow_pickle=True)
data_test = np.load("../data/training_2c.npy", allow_pickle=True)
print(f"Number of samples: {data_train.shape[1]} / {data_test.shape[1]}")

train_set = Connect4Dataset(data_train[0], data_train[1], data_train[2], training=True)
test_set = Connect4Dataset(data_test[0], data_test[1], data_test[2], training=False)


args = TrainingArgs(
    train_epochs=8,
    batch_size=500,
    print_progress=True,
    # from_pretrained="../models/model_0.pth",
    model_output_path="../models/model_2.pth"
)
trainer = Trainer(model=NaiveNet(ROW_COUNT, COLUMN_COUNT),
                  train_dataset=train_set,
                  test_dataset=test_set,
                  training_args=args
                  )

trainer.train()
