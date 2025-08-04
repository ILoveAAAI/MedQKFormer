import os
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from utils.utils import calculate_metrics
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn.functional as F
from spikingjelly.clock_driven import functional

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'dsc': 0.0,
        'acc': 0.0,
        'sen': 0.0,
        'spe': 0.0,
        'pre': 0.0,
        'rec': 0.0,
        'fb': 0.0,
        'em': 0.0
    }

    for i, (x, y) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        functional.reset_net(model)
        
        mask_pred = model(x)
        loss=loss_fn(mask_pred, y)
        
        loss.backward()
        optimizer.step()

        metrics['loss'] += loss.item()

def evaluate(model, loader, loss_fn, device):
    model.eval()

    metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'dsc': 0.0,
        'acc': 0.0,
        'sen': 0.0,
        'spe': 0.0,
        'pre': 0.0,
        'rec': 0.0,
        'fb': 0.0,
        'em': 0.0
    }

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            functional.reset_net(model)
            mask_pred = model(x)
            loss=loss_fn(mask_pred, y)

            metrics['loss'] += loss.item()

            batch_metrics = {
                'miou': [], 'dsc': [], 'acc': [], 'sen': [],
                'spe': [], 'pre': [], 'rec': [], 'fb': [], 'em': []
            }

            for yt, yp in zip(y, mask_pred):
                scores = calculate_metrics(yt, yp)
                for idx, key in enumerate(batch_metrics.keys()):
                    batch_metrics[key].append(scores[idx])

            for key in batch_metrics:
                metrics[key] += np.mean(batch_metrics[key])

    for key in metrics:
        metrics[key] /= len(loader)

    return metrics['loss'], [
        metrics['miou'], metrics['dsc'], metrics['acc'], 
        metrics['sen'], metrics['spe'], metrics['pre'],
        metrics['rec'], metrics['fb'], metrics['em']
    ]

