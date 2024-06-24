from lightning_fabric.utilities.device_dtype_mixin import override
import numpy as np
import torch
import pandas as pd
import os
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from CustomDataset import CustomDataset

class MalClassifierForFSL(pl.LightningModule):
  def __init__(self, hyper_parameter: dict):
    super().__init__()
    
    #파라미터
    self.MAX_LENGTH = hyper_parameter["max_length"] if ("max_length" in hyper_parameter) else 400
    self.LEARNING_RATE = hyper_parameter["lr"] if ("lr" in hyper_parameter) else 1e-6
    self.EPOCHS = hyper_parameter["epochs"] if ("epochs" in hyper_parameter) else 20
    self.OPTIMIZER = hyper_parameter["optimizer"] if ("optimizer" in hyper_parameter) else "adamw"
    self.GAMMA = hyper_parameter["gamma"] if ("gamma" in hyper_parameter) else 0.5
    self.BATCH_SIZE = hyper_parameter["batch_size"] if ("batch_size" in hyper_parameter) else 32
    self.DATAFRAME = hyper_parameter["DataFrame"] if ("DataFrame" in hyper_parameter) else pd.read_csv('./data_label_withBenign.csv')
    self.CLASS_NUM = hyper_parameter["class_num"] if ("class_num" in hyper_parameter) else 5
    self.STATIC_PATH = hyper_parameter["static_path"] if ("static_path" in hyper_parameter) else "../Integrated System/Data/prepro/datanumpy/images/"
    self.DYNAMIC_PATH = hyper_parameter["dynamic_path"] if ("dynamic_path" in hyper_parameter) else "../Integrated System/Data/prepro/datanumpy/tfidf/"
    self.IMG_TYPE = hyper_parameter["img_type"] if ("img_type" in hyper_parameter) else "orig"
    
    # 데이터 셋
    self._traindata =  self.DATAFRAME.loc[ self.DATAFRAME['train']=='train']
    self._valdata =  self.DATAFRAME.loc[ self.DATAFRAME['train']=='val']
    self._testdata =  self.DATAFRAME.loc[ self.DATAFRAME['train']=='test']
    
    self.image_paths = []
    self.tfidf_paths = []
    self.class_labels = []

    self.CNN_STATIC = torch.nn.Sequential(
        torch.nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
    )
    
    self.LSTM_DYNAMIC = torch.nn.Sequential(
        torch.nn.LSTM(input_size=self.MAX_LENGTH, hidden_size=120,
                      num_layers=2, batch_first=True, bidirectional=True),
    )
    
    self.relu = torch.nn.ReLU()
    self.drop = torch.nn.Dropout(p=0.5)

    self.d2 = torch.nn.Linear(4464, 256)
    self.d3 = torch.nn.Linear(256, 128)
    self.d4 = torch.nn.Linear(128, 64)
    # Fully connected 1 (readout)
    self.d5 = torch.nn.Linear(64, self.CLASS_NUM)
    self.softmax = torch.nn.Softmax(dim=1)
    self.lossF = torch.nn.CrossEntropyLoss()
        
        
        
  def forward(self, dy, st):
    x2,_ = self.LSTM_DYNAMIC(dy)
    x2 = x2.reshape(x2.size(0), -1)
    x1 = self.CNN_STATIC(st)
    x1 = x1.reshape(x1.size(0), -1)

    # Concatenate in dim1 (feature dimension)
    out = torch.cat((x1, x2), 1)
    out = self.drop(out)
    out = self.d2(out)
    out = self.relu(out)
    out = self.d3(out)
    out = self.relu(out)
    out = self.d4(out)
    out = self.relu(out)
    out = self.d5(out)
    out = self.softmax(out)
    return out
    
  def __dataloader(self, image_paths, tfidf_paths, class_labels, shuffle: bool = False):
    dataset = CustomDataset(image_paths, tfidf_paths, class_labels)
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=self.BATCH_SIZE
    )
    
  def __preprocessing(self, data_list):
    for idx, i in data_list.iterrows():
        datanumpy_tfidf = str(i['hash'])+'.npy'
        datanumpy_img = str(i['hash'])+'.png'
        self.image_paths += [os.path.join(self.STATIC_PATH, self.IMG_TYPE, datanumpy_img)]
        self.tfidf_paths += [os.path.join(self.DYNAMIC_PATH, str(self.MAX_LENGTH), datanumpy_tfidf)]
        
        self.class_labels += [i['label']]
    return self.image_paths, self.tfidf_paths, self.class_labels

  def train_dataloader(self):
    self.image_paths, self.tfidf_paths, self.class_labels = self.__preprocessing(self._traindata)
    return self.__dataloader(self.image_paths, self.tfidf_paths, self.class_labels, True)

  def val_dataloader(self):
    self.image_paths, self.tfidf_paths, self.class_labels = self.__preprocessing(self._valdata)
    return self.__dataloader(self.image_paths, self.tfidf_paths, self.class_labels)

  def test_dataloader(self):
    self.image_paths, self.tfidf_paths, self.class_labels = self.__preprocessing(self._testdata)
    return self.__dataloader(self.image_paths, self.tfidf_paths, self.class_labels)
    
  def __step(self, batch, batch_idx):
    image_paths, tfidf_paths, class_labels = batch
    output = self.forward(dy=tfidf_paths, st=image_paths)

    logits = output
    loss = self.lossF(logits, class_labels)

    preds = logits.argmax(dim=-1)
    
    logs = {'train_loss': loss}

    return {'loss': loss, 'log': logs}
  
  def training_step(self, batch, batch_idx):
    return self.__step(batch, batch_idx)

  def validation_step(self, batch, batch_idx):
    return self.__step(batch, batch_idx)

  def test_step(self, batch, batch_idx):
    return self.__step(batch, batch_idx)
    
  def __epoch_end(self, outputs, state="train"):
    loss = torch.tensor(0, dtype=torch.float64)
    y_true, y_pred = [], []

    for i in outputs:
        loss += i["loss"].cpu().detach()
        y_true += i["y_true"]
        y_pred += i["y_pred"]

    loss = loss / len(outputs)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"[Epoch {self.trainer.current_epoch} {state.upper()}]",
          f"Loss={loss}, Acc={acc}", "CM={}".format(str(cm).replace("\n", "")))

    return {"loss": loss, "acc": acc}
              
  def configure_optimizers(self):
    if self.OPTIMIZER == "adam":
        optimizer = Adam(self.parameters(), lr=self.LEARNING_RATE)
    elif self.OPTIMIZER == "adamw":
        optimizer = AdamW(self.parameters(), lr=self.LEARNING_RATE)
    elif self.OPTIMIZER == "sgd":
        optimizer = SGD(self.parameters(), lr=self.LEARNING_RATE)
    else:
        raise NotImplementedError(f"'{self.OPTIMIZER}' is not available.")

    scheduler = ExponentialLR(optimizer, gamma=self.GAMMA)

    return {
        "optimizer": optimizer,
        "scheduler": scheduler
    }

  def _on_train_epoch_end(self, outputs):
    self.__epoch_end(outputs, state="train")

  def _on_validation_epoch_end(self, outputs):
    self.__epoch_end(outputs, state="val")

  def _on_test_epoch_end(self, outputs):
    self.__epoch_end(outputs, state="test")