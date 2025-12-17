"""
Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
Code for reproducing results on BCI Competition 2020 Track #3: Imagined Speech Classification.
Currently under review for publication.
Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import torch
torch.set_num_threads(8)
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchmetrics
import logging
import h5py
import einops
from sklearn.model_selection import KFold
from transformers import PretrainedConfig
import lightning as pl
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

from FAST import FAST
from FAST_mamba import FAST_Mamba2
from utils import green, yellow
from BCIC2020Track3_preprocess import Electrodes, Zones

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def load_standardized_h5(cache_fn):
    X, Y = [], []
    with h5py.File(cache_fn, 'r') as f:
        subjects = list(f.keys())
        for subject in subjects:
            X.append(f[subject]['X'][()])
            Y.append(f[subject]['Y'][()])
    X, Y = np.array(X), np.array(Y)
    print('Loaded from', cache_fn, X.shape, Y.shape)
    return X, Y

def inference_on_loader(model, loader):
    model.eval()
    # Detect device from the model parameters (works for cpu, cuda, mps)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        Pred, Real = [], []
        for x, y in loader:
            # Move input to the same device as the model
            x = x.to(device)
            
            # Forward pass
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu()
            
            Pred.append(preds)
            Real.append(y)
        Pred, Real = torch.cat(Pred), torch.cat(Real)
    return Pred.numpy(), Real.numpy()

class BasicDataset(Dataset):
    def __init__(self, data, label):
        if len(data.shape) == 4:
            data, label = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
        self.data, self.labels = torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        return sample, label

class EEG_Encoder_Module(pl.LightningModule):
    def __init__(self, config, max_epochs, niter_per_ep, model_type):
        super().__init__()
        self.config = config
        if model_type == 'transformer':
            self.model = FAST(config)
        elif model_type == 'mamba2':
            self.model = FAST_Mamba2(config)
        self.loss = nn.CrossEntropyLoss()
        self.cosine_lr_list = cosine_scheduler(1, 0.1, max_epochs, niter_per_ep, warmup_epochs=10)
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes = config.n_classes)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: self.cosine_lr_list[self.global_step-1])
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return self.loss(pred, y)

def Finetune(args, config, Data_X, Data_Y, logf, max_epochs=200, ckpt_pretrain=None, model_type='transformer'):
    seed_all(42)
    Pred, Real = [], []
    kf = KFold(n_splits=5, shuffle=False)
    for _train_idx, _test_idx in kf.split(Data_X):
        x_train, y_train = Data_X[_train_idx], Data_Y[_train_idx]
        x_test, y_test = Data_X[_test_idx], Data_Y[_test_idx]

        train_data = BasicDataset(x_train, y_train)
        train_loader = DataLoader(train_data, batch_size=len(x_train), shuffle=True, num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)
        test_data = BasicDataset(x_test, y_test)
        test_loader = DataLoader(test_data, batch_size=len(x_test), shuffle=False, num_workers=os.cpu_count(), persistent_workers=True, pin_memory=True)

        model = EEG_Encoder_Module(config, max_epochs, len(train_loader), model_type)
        if ckpt_pretrain is not None:
            model.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

        print(yellow(logf), green(ckpt_pretrain), x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        if args.accelerator == 'gpu':
            trainer = pl.Trainer(strategy='auto', accelerator='gpu', devices=[args.gpu], max_epochs=max_epochs, callbacks=[], 
                            enable_progress_bar=True, enable_checkpointing=False, precision='bf16-mixed', logger=False)
        elif args.accelerator == 'mps':
            trainer = pl.Trainer(strategy='auto', accelerator='mps', devices=1, max_epochs=max_epochs, callbacks=[], 
                            enable_progress_bar=True, enable_checkpointing=False, precision='bf16-mixed', logger=False)
        trainer.fit(model, train_dataloaders=train_loader)

        # Test data is used only once
        pred, real = inference_on_loader(model.model, test_loader)
        Pred.append(pred)
        Real.append(real)
    Pred, Real = np.concatenate(Pred), np.concatenate(Real)
    np.savetxt(logf, np.array([Pred, Real]).T, delimiter=',', fmt='%d')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--gpu', type=int, default=0)
    # default to mps training because I use a macbook 
    # but switch to gpu whenever Nvidia GPU is available
    args.add_argument('--accelerator', type=str, default='mps')
    args.add_argument('--folds', type=str, default='0-15')
    args.add_argument('--model', choices=['transformer', 'mamba2'], default='transformer')
    args.add_argument('--use_spatial_projection', type=str2bool, default=True)
    args = args.parse_args()

    if '-' in args.folds:
        start, end = [int(x) for x in args.folds.split('-')]
        args.folds = list(range(start, end))
    else:
        args.folds = [int(x) for x in args.folds.split(',')]

    Run = f"Results/FAST-{args.model}-spatial_projection-{args.use_spatial_projection}/"
    os.makedirs(f"{Run}", exist_ok=True)

    sfreq = 250
    if args.model == 'transformer':
        config = PretrainedConfig(
            electrodes=Electrodes,
            zone_dict=Zones,
            dim_cnn=32,
            dim_token=32,
            seq_len=800,
            window_len=sfreq,
            slide_step=sfreq//2,
            head='Conv4Layers',
            n_classes=5,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            use_spatial_projection=args.use_spatial_projection,
        )
    elif args.model == 'mamba2':
        config = PretrainedConfig(
            electrodes=Electrodes,
            zone_dict=Zones,
            dim_cnn=32,
            dim_token=32,
            seq_len=800,
            window_len=sfreq,
            slide_step=sfreq // 2,
            head="Conv4Layers",
            n_classes=5,
            num_layers=4,   # number of Mamba2 blocks
            num_heads=8,    # kept for compatibility (unused)
            dropout=0.1,
            # Mamba2 hyperparams
            mamba_d_state=64,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_headdim=64,
            mamba_ngroups=1,
            use_spatial_projection=args.use_spatial_projection,
        )
        
    X, Y = load_standardized_h5('Processed/BCIC2020Track3.h5')
    for fold in range(15):
        if fold not in args.folds:
            continue
        flog = f"{Run}/{fold}-Tune.csv"
        if os.path.exists(flog):
            print(f"Skip {flog}")
            continue
        Finetune(args, config, X[fold], Y[fold], flog, max_epochs=200, model_type=args.model)

    accuracies = {}
    for fold in range(15):
        flog = f"{Run}/{fold}-Tune.csv"
        if not os.path.exists(flog):
            print(f"Skip {flog}")
            continue
        data = np.loadtxt(flog, delimiter=',', dtype=int)
        pred, label = data[:, 0], data[:, 1]
        accuracies[fold] = np.mean(pred == label)

    if accuracies:
        acc_data = np.array([[k, accuracies[k]] for k in sorted(accuracies.keys())])
        np.savetxt(f"{Run}/accuracies.csv", acc_data, delimiter=',', header="Fold,Accuracy", fmt=['%d', '%.4f'], comments='')
    
    return args.model, args.use_spatial_projection, accuracies 

if __name__ == '__main__':
    model, use_spatial_projection, accuracies = main()
    print(f"Model: {model}, Use Spatial Projection: {use_spatial_projection}")
    
    acc_list = []
    for fold in sorted(accuracies.keys()):
        acc = accuracies[fold]
        acc_list.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    print(f"Accuracy: {np.mean(acc_list):.4f}, Std: {np.std(acc_list):.4f}")