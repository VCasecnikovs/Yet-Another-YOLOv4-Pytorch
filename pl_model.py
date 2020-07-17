import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import ListDataset
from model import YOLOv4

from lars import LARS


class YOLOv4PL(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.train_ds = ListDataset(hparams.train_ds, train=True)
        self.valid_ds = ListDataset(hparams.valid_ds, train=False)

        self.model = YOLOv4(n_classes = 5, pretrained=True).cuda()

    def train_dataloader(self):
        train_dl = DataLoader(self.train_ds, batch_size=self.hparams.bs, collate_fn=self.train_ds.collate_fn, pin_memory=True) 
        return train_dl
    
    def val_dataloader(self):
        valid_dl = DataLoader(self.valid_ds, batch_size=self.hparams.bs, collate_fn=self.valid_ds.collate_fn, pin_memory=True)
        return valid_dl

    def forward(self, x, y=None):
        return self.model(x, y)

    def basic_training_step(self, batch):
        filenames, images, labels = batch
        y_hat, loss = self(images, labels)

        logger_logs = {"training_loss" : loss}

        return {"loss" : loss, "log" : logger_logs}

    def sat_training_step(self, batch, epsilon = 0.01):
        filenames, images, labels = batch

        images.requires_grad_(True)
        y_hat, loss = self(images, labels)
        loss.backward()
        data_grad = images.grad.data
        images.requires_grad_(False)
        images = torch.clamp(images + data_grad.sign() * epsilon, 0, 1)

        return self.basic_training_step((filenames, images, labels))

        


    def training_step(self, batch, batch_idx):
        if hparams.SAT:
            pass
        else:
            return self.basic_training_step(batch)
        
        

    def training_epoch_end(self, outputs):
        training_loss_mean = torch.stack([x['training_loss'] for x in outputs]).mean()
        return {"loss" : training_loss_mean, "log" : {"training_loss_epoch" : training_loss_mean}}

    def validation_step(self, batch, batch_idx):
        filenames, images, labels = batch
        y_hat, loss = self(images, labels)
        return {"val_loss" : loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logger_logs = {"validation_loss" : val_loss_mean}

        return {"val_loss" : val_loss_mean, "log" : logger_logs}

    def configure_optimizers(self):
        #With this thing we get only params, which requires grad (weights needed to train)
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.hparams.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(params, self.hparams.lr, momentum = self.hparams.momentum, weight_decay=self.hparams.wd)
        elif self.hparams.optimizer == "LARS":
            self.optimizer = LARS(params, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.wd, max_epoch=self.hparams.epochs)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.hparams.lr, epochs=self.hparams.epochs, steps_per_epoch=1, pct_start=self.hparams.pct_start)
        sched_dict = {'scheduler': self.scheduler}

        return [self.optimizer], [sched_dict]

