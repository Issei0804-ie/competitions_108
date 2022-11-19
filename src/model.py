import pytorch_lightning as pl
import torch
import torchmetrics
from torchvision.models import resnet18


class ImageClassifierModel(torch.nn.Module):
    def __init__(self, lr: float, num_group: int):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        num_last_features = self.resnet.fc.out_features
        self.regressor = torch.nn.Linear(num_last_features, num_group)
        self.lr = lr

    def forward(self, image):
        outputs = self.resnet(image)
        logits = self.regressor(outputs)
        return logits


class TrainModel(pl.LightningModule):
    def __init__(self, lr: float, num_group: int):
        super().__init__()
        self.model = ImageClassifierModel(lr, num_group)
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch[0])

    def training_step(self, batch, batch_idx):
        output = self.model(batch[0])
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(output, batch[1])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch[0])
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(output, batch[1])
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        output = self.model(batch[0])
        cross_entropy = torch.nn.CrossEntropyLoss()
        #loss = cross_entropy(output, batch[1])
        #self.log("test_loss", loss)
        #predicted = torch.argmax(output, dim=1)
        #acc = torchmetrics.Accuracy().to(device="cuda")
        #value = acc(predicted, batch[1])
        #self.log("test_accuracy", value)

    def configure_optimizers(self):
        opti = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return opti
