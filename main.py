import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from model import create_resnet50
from dataloader import DataLoaderFactory
from trainer import Trainer
from collections import Counter
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Prevent numerical instability by using -ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Function to calculate class frequencies
def calculate_class_frequencies(dataloader):
    """
    Calculate normalized inverse class frequencies for FocalLoss using the training dataloader.

    Returns:
        norm_inv_class_frequencies (list): Normalized inverse class frequencies.
    """
    # Extract all labels from the training dataloader
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())
    
    # Count the occurrences of each class
    class_counts = Counter(all_labels)
    
    # Calculate frequencies and inverse frequencies
    total_samples = sum(class_counts.values())
    class_frequencies = {cls: count / total_samples for cls, count in class_counts.items()}
    inv_class_frequencies = {cls: total_samples / count for cls, count in class_counts.items()}
    norm_inv_class_frequencies = [inv_class_frequencies[cls] / sum(inv_class_frequencies.values()) for cls in sorted(class_counts.keys())]
    
    return norm_inv_class_frequencies

def main(config):
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data loading
    data_loader_factory = DataLoaderFactory(config['data_dir'], config['batch_size'], config['num_workers'],  config['mixup_option'], config['mixup_alpha'])
    mean, std = data_loader_factory.compute_mean_std()
    dataloaders, _, class_names = data_loader_factory.create_dataloaders(mean, std)

    # Model setup
    model = create_resnet50(pretrained_source=config['pretrained_source'], num_classes=config['num_classes'], num_fc_layers=config['num_fc_layers'])
    model = model.to(device)

    # Criterion (Loss Function) setup
    if config['criterion'] == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif config['criterion'] == 'focal_loss':
        alpha = torch.tensor(calculate_class_frequencies(dataloaders['train'])).to(device) if config.get('alpha_option') == 'inverse_freq' else None
        criterion = FocalLoss(alpha=alpha, gamma=config.get('gamma', 2), reduction='mean')
    else:
        raise ValueError("Unsupported criterion specified")

    # Optimizer setup
    optimizer = optim.Adam(model.fc.parameters(), lr=config['learning_rate'])
    
    # Scheduler setup
    if config['scheduler_type'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif config['scheduler_type'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    else:
        scheduler = None  # If no scheduler is specified

    # Initialize and run the trainer
    trainer = Trainer(model, dataloaders, criterion, optimizer, scheduler, device, selection_metric=config['selection_metric'], num_epochs=config['num_epochs'], mixup_option=config['mixup_option'], mixup_alpha=config['mixup_alpha'], data_loader_factory=data_loader_factory)
    model = trainer.train()

    # Evaluate the model
    trainer.evaluate(dataloaders['test'], class_names)

def main_visualise_mixup(config):
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data loading
    data_loader_factory = DataLoaderFactory(config['data_dir'], config['batch_size'], config['num_workers'], config['mixup_option'], config['mixup_alpha'])
    mean, std = data_loader_factory.compute_mean_std()
    dataloaders, _, class_names = data_loader_factory.create_dataloaders(mean, std)

    # Get a batch of training images and labels
    images, labels = next(iter(dataloaders['train']))

    # Visualize Mixup (either standard or balanced, based on the mixup_option attribute)
    data_loader_factory.visualize_mixup(images, labels, mean, std)

if __name__ == "__main__":
    # Example configuration dictionary
    config = {
        'seed': 42,
        'data_dir': '/cs/student/projects3/aibh/2023/jingqzhu/data/colonoscopic/frames',
        'batch_size': 32,
        'num_workers': 4,
        'pretrained_source': 'gastronet',  # or 'gastronet' or 'imagenet'
        'num_classes': 3,
        'num_fc_layers': 1,  # 1 or 2
        'learning_rate': 1e-3,
        'scheduler_type': 'cosine',  # 'cosine' or 'step'
        'selection_metric': 'mcc',  # 'f1', 'bacc', or 'mcc'
        'num_epochs': 30,
        'criterion': 'cross_entropy',  # 'cross_entropy' or 'focal_loss'
        'alpha_option': None,  # 'inverse_freq' or None
        'gamma': 5,  # Focusing parameter for Focal Loss
        'mixup_option': 'balanced_mixup', # 'none', 'mixup', 'balanced_mixup'
        'mixup_alpha': 0.1
    }
    main(config)
    # main_visualise_mixup(config)
    