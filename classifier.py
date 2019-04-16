import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import cv2
import torch
from tqdm import tqdm_notebook
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from torchvision import transforms

%matplotlib inline

def kaggle_commit_logger(str_to_log, need_print = True):
    if need_print:
        print(str_to_log)
    os.system('echo ' + str_to_log)

def cuda(x):
    return x.cuda(non_blocking=True)

def f1_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


def validate(model, valid_loader, criterion, need_tqdm=False):
    model.eval();

    test_loss = 0.0
    TH_TO_ACC = 0.5

    true_ans_list = []
    preds_cat = []

    with torch.no_grad():

        if need_tqdm:
            valid_iterator = tqdm_notebook(valid_loader)
        else:
            valid_iterator = valid_loader

        for step, (features, targets) in enumerate(valid_iterator):
            features, targets = cuda(features), cuda(targets)

            logits = model(features)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            true_ans_list.append(targets)
            preds_cat.append(torch.sigmoid(logits))

        all_true_ans = torch.cat(true_ans_list)
        all_preds = torch.cat(preds_cat)

        f1_eval = f1_score(all_true_ans, all_preds).item()

    logstr = f'Mean val f1: {round(f1_eval, 5)}'
    kaggle_commit_logger(logstr)
    return test_loss / (step + 1), f1_eval


def get_subm_answers(model, subm_dataloader, need_tqdm=False):
    model.eval();
    preds_cat = []
    ids = []

    with torch.no_grad():

        if need_tqdm:
            subm_iterator = tqdm_notebook(subm_dataloader)
        else:
            subm_iterator = subm_dataloader

        for step, (features, subm_ids) in enumerate(subm_iterator):
            features = cuda(features)

            logits = model(features)
            preds_cat.append(torch.sigmoid(logits))
            ids += subm_ids

        all_preds = torch.cat(preds_cat)
        all_preds = torch.argmax(all_preds, dim=1).int().cpu().numpy()
    return all_preds, ids

def process_one_id(id_classes_str):
    if id_classes_str:
        return REVERSE_CLASSMAP[int(id_classes_str)]
    else:
        return id_classes_str


def train_one_epoch(model, train_loader, criterion, optimizer, steps_upd_logging=250):
    model.train();

    total_loss = 0.0

    train_tqdm = tqdm_notebook(train_loader)

    for step, (features, targets) in enumerate(train_tqdm):
        features, targets = cuda(features), cuda(targets)

        optimizer.zero_grad()

        logits = model(features)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            logstr = f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}'
            train_tqdm.set_description(logstr)
            kaggle_commit_logger(logstr, need_print=False)

    return total_loss / (step + 1)

class IMetDataset(Dataset):

    def __init__(self,
                 df,
                 images_dir,
                 n_classes=NUM_CLASSES,
                 id_colname=ID_COLNAME,
                 answer_colname=ANSWER_COLNAME,
                 label_dict=CLASSMAP,
                 transforms=None
                 ):
        self.df = df
        self.images_dir = images_dir
        self.n_classes = n_classes
        self.id_colname = id_colname
        self.answer_colname = answer_colname
        self.label_dict = label_dict
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_name = img_id  # + self.img_ext
        img_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.answer_colname is not None:
            label = torch.zeros((self.n_classes,), dtype=torch.float32)
            label[self.label_dict[cur_idx_row[self.answer_colname]]] = 1.0

            return img, label

        else:
            return img, img_id
