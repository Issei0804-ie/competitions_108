import argparse
import logging
import os.path
import warnings

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader, Subset

from src.dataset_builder import TestDataset
from src.model import TrainModel

if __name__ == "__main__":
    logging.getLogger("pytorch_lightning").setLevel(logging.NOTSET)
    logging.getLogger("torch.utils.data").setLevel(logging.NOTSET)
    logging.getLogger("transformers").setLevel(logging.NOTSET)
    warnings.simplefilter("ignore")

    p = argparse.ArgumentParser(description="学習したモデルを使用して破綻検出を行うツール")
    p.add_argument("checkpoint")
    p.add_argument("data_dir")
    args = p.parse_args()
    cptk = args.checkpoint
    data_dir = args.data_dir

    trainer = pl.Trainer(accelerator="gpu", devices=1)

    model = TrainModel(0.0001, 20)

    dataset = TestDataset([data_dir])
    test_loader = DataLoader(dataset, batch_size=32 * 5, num_workers=os.cpu_count())
    hoge = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=cptk)
    print(len(hoge))
    result = []
    for fuga in hoge:
        result.extend(torch.softmax(fuga, -1).tolist())

    for i in range(10000):
        print(
            f"test_{str(i).zfill(5)}.png,{str(result[i]).replace('[', '').replace(']', '')}"
        )
