import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import ListDataset
from model import YOLOv4

from lars import LARS
from ranger import Ranger
from radam import RAdam

from sched_del import DelayedCosineAnnealingLR

torch.backends.cudnn.benchmark = True

class YOLOv4PL(pl.LightningModule):
    def __init__(self, hparams):
        """
        Initialize all the loss.

        Args:
            self: (todo): write your description
            hparams: (dict): write your description
        """
        super().__init__()

        self.hparams = hparams

        self.train_ds = ListDataset(hparams.train_ds, train=True)
        self.valid_ds = ListDataset(hparams.valid_ds, train=False)

        self.model = YOLOv4(n_classes=5,
            pretrained=hparams.pretrained,
            dropblock=hparams.Dropblock,
            sam=hparams.SAM,
            eca=hparams.ECA,
            ws=hparams.WS,
            iou_aware=hparams.iou_aware,
            coord=hparams.coord,
            hard_mish=hparams.hard_mish,
            asff=hparams.asff,
            repulsion_loss=hparams.repulsion_loss,
            acff=hparams.acff,
            bcn=hparams.bcn,
            mbn=hparams.mbn).cuda()

    def train_dataloader(self):
        """
        Parameters ----------

        Args:
            self: (todo): write your description
        """
        train_dl = DataLoader(self.train_ds, batch_size=self.hparams.bs, collate_fn=self.train_ds.collate_fn, pin_memory=True, num_workers=4)
        return train_dl

    def val_dataloader(self):
        """
        Validate dataler.

        Args:
            self: (todo): write your description
        """
        valid_dl = DataLoader(self.valid_ds, batch_size=self.hparams.bs, collate_fn=self.valid_ds.collate_fn, pin_memory=True, num_workers=4)
        return valid_dl

    def forward(self, x, y=None):
        """
        Forward function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            y: (todo): write your description
        """
        return self.model(x, y)

    def basic_training_step(self, batch):
        """
        Perform training step.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
        """
        filenames, images, labels = batch
        y_hat, loss = self(images, labels)
        logger_logs = {"training_loss": loss}

        return {"loss": loss, "log": logger_logs}

    def sat_fgsm_training_step(self, batch, epsilon=0.01):
        """
        Perform an optimization step.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
            epsilon: (float): write your description
        """
        filenames, images, labels = batch

        images.requires_grad_(True)
        y_hat, loss = self(images, labels)
        loss.backward()
        data_grad = images.grad.data
        images.requires_grad_(False)
        images = torch.clamp(images + data_grad.sign() * epsilon, 0, 1)
        return self.basic_training_step((filenames, images, labels))

    def sat_vanila_training_step(self, batch, epsilon=1):
        """
        Perform a forward step.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
            epsilon: (float): write your description
        """
        filenames, images, labels = batch

        images.requires_grad_(True)
        y_hat, loss = self(images, labels)
        loss.backward()
        data_grad = images.grad.data
        images.requires_grad_(False)
        images = torch.clamp(images + data_grad, 0, 1)
        return self.basic_training_step((filenames, images, labels))
        


    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
            batch_idx: (str): write your description
        """
        if self.hparams.SAT == "vanila":
            return self.sat_vanila_training_step(batch, self.hparams.epsilon)
        elif self.hparams.SAT == "fgsm":
            return self.sat_fgsm_training_step(batch, self.hparams.epsilon)
        else:
            return self.basic_training_step(batch)

    def training_epoch_end(self, outputs):
        """
        Training function.

        Args:
            self: (todo): write your description
            outputs: (todo): write your description
        """
        training_loss_mean = torch.stack([x['training_loss'] for x in outputs]).mean()
        return {"loss": training_loss_mean, "log": {"training_loss_epoch": training_loss_mean}}

    def validation_step(self, batch, batch_idx):
        """
        Runs the validation step.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
            batch_idx: (int): write your description
        """
        filenames, images, labels = batch
        y_hat, loss = self(images, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Compute the loss loss.

        Args:
            self: (todo): write your description
            outputs: (todo): write your description
        """
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logger_logs = {"validation_loss": val_loss_mean}

        return {"val_loss": val_loss_mean, "log": logger_logs}

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Args:
            self: (todo): write your description
        """
        # With this thing we get only params, which requires grad (weights needed to train)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.hparams.optimizer == "Ranger":
            self.optimizer = Ranger(params, self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(params, self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.wd)
        elif self.hparams.optimizer == "LARS":
            self.optimizer = LARS(params, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.wd, max_epoch=self.hparams.epochs)
        elif self.hparams.optimizer == "RAdam":
            self.optimizer = RAdam(params, lr=self.hparams.lr, weight_decay=self.hparams.wd)

        if self.hparams.scheduler == "Cosine Warm-up":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.hparams.lr, epochs=self.hparams.epochs, steps_per_epoch=1, pct_start=self.hparams.pct_start)
        if self.hparams.scheduler == "Cosine Delayed":
            self.scheduler = DelayedCosineAnnealingLR(self.optimizer, self.hparams.flat_epochs, self.hparams.cosine_epochs)

        
        sched_dict = {'scheduler': self.scheduler}


        return [self.optimizer], [sched_dict]
