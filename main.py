import argparse
import csv
import datetime
import os.path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import src.dataset_builder
from src.model import TrainModel

filepath2label = {}

p = argparse.ArgumentParser(description="学習時に使用するスクリプト")
p.add_argument("log_name")
args = p.parse_args()
log_name = args.log_name

with open("train_master.tsv", mode="r") as fh:
    reader = csv.reader(fh, delimiter="\t")
    filepath2label = {row[0]: row[1] for row in reader}

dataset = src.dataset_builder.LabeledDataset(["train"], filepath2label)

n_samples = len(dataset)
train_size = int(len(dataset) * 0.6)
val_size = int(len(dataset) * 0.2)
test_size = n_samples - (train_size + val_size)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32 * 1, num_workers=os.cpu_count(), shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32 * 1, num_workers=os.cpu_count()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32 * 1, num_workers=os.cpu_count()
)
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="model/",
    filename=f"{str(datetime.datetime.today())}",
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

logger = TensorBoardLogger(save_dir=".", version=log_name)

trainer = pl.Trainer(
    logger=logger,
    gpus=1,
    max_epochs=100,
    callbacks=[
        checkpoint,
        pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min"),
        lr_monitor,
    ],
)

model = TrainModel(lr=1e-5, num_group=20)
trainer.fit(model, train_loader, val_loader)

test = trainer.test(model, test_loader)
print(test)
