import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt

class DataLoaderFactory:
    def __init__(self, data_dir, batch_size=32, num_workers=4, mixup_option='none', mixup_alpha=1.0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_option = mixup_option
        self.mixup_alpha = mixup_alpha

    def compute_mean_std(self):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        train_data_dir = os.path.join(self.data_dir, 'train')
        dataset = datasets.ImageFolder(train_data_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        nb_samples = 0
        
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            nb_samples += batch_samples
        
        mean /= nb_samples
        std /= nb_samples
        
        return mean, std

    def mixup_data(self, x, y):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, index

    def balanced_mixup_data(self, x, y):
        """Balanced Mixup: returns mixed inputs, pairs of targets, and lambda."""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, 1)
        else:
            lam = 1

        batch_size = x.size(0)
        mixed_x = torch.zeros_like(x)
        y_a, y_b = torch.zeros_like(y), torch.zeros_like(y)
        index = torch.zeros(batch_size, dtype=torch.long)

        # Identify unique classes in the batch
        unique_classes = torch.unique(y)

        for i in range(batch_size):
            # First Sample: Sample based on class frequency in the batch
            x_i = x[i]
            y_i = y[i]

            # Second Sample: Sample with equal probability across classes
            chosen_class = random.choice(unique_classes.tolist())
            class_indices = (y == chosen_class).nonzero(as_tuple=True)[0]
            j = random.choice(class_indices)
            x_j = x[j]
            y_j = y[j]

            # Mix the inputs and labels
            mixed_x[i] = lam * x_i + (1 - lam) * x_j
            y_a[i] = y_i
            y_b[i] = y_j
            index[i] = j  # Store the index used

        return mixed_x, y_a, y_b, lam, index

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def create_dataloaders(self, mean, std, augmentations=None):
        if augmentations is None:
            augmentations = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),  
                    transforms.RandomRotation(360),      
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),  
                    transforms.RandomChoice([transforms.RandomPerspective(distortion_scale=0.25), 
                                             transforms.RandomAffine(degrees=(-45, 45), translate=(0, 0.0625), scale=(1.0, 1.05))]),
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]),
                'test': transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]),
            }

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), augmentations[x]) for x in ['train', 'val', 'test']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=(x == 'train'), num_workers=self.num_workers) for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes

        return dataloaders, dataset_sizes, class_names
    
    def denormalize(self, img, mean, std):
        """
        Denormalizes a tensor image.
        Args:
            img (torch.Tensor): Image tensor to be denormalized.
            mean (tuple): Mean used for normalization.
            std (tuple): Standard deviation used for normalization.
        Returns:
            torch.Tensor: Denormalized image tensor.
        """
        if isinstance(mean, torch.Tensor):
            mean = mean.clone().detach().view(3, 1, 1)
        else:
            mean = torch.tensor(mean).clone().detach().view(3, 1, 1)

        if isinstance(std, torch.Tensor):
            std = std.clone().detach().view(3, 1, 1)
        else:
            std = torch.tensor(std).clone().detach().view(3, 1, 1)
        img = img * std + mean  # Reverses the normalization process
        return img

    def visualize_mixup(self, x, y, mean, std):
        """
        Visualize the effect of Mixup by displaying original and mixed images.

        Args:
            x (torch.Tensor): Batch of images.
            y (torch.Tensor): Batch of labels.
        """
        if self.mixup_option == 'balanced_mixup':
            mixed_x, y_a, y_b, lam, index = self.balanced_mixup_data(x, y)
        elif self.mixup_option == 'mixup':
            mixed_x, y_a, y_b, lam, index = self.mixup_data(x, y)
        else:
            raise ValueError("Mixup visualization requires 'mixup' or 'balanced_mixup' option.")

        x = self.denormalize(x, mean, std)
        mixed_x = self.denormalize(mixed_x, mean, std)

        # Convert tensors to numpy arrays for visualization
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        mixed_x_np = mixed_x.permute(0, 2, 3, 1).cpu().numpy()

        # Plot original images A, B, and the mixed images
        batch_size = x_np.shape[0]
        num_mix_pair = 4
        fig, axes = plt.subplots(3, num_mix_pair, figsize=(num_mix_pair * 5, 15))
        #fig.suptitle(f'Mixup Visualization (Lambda = {lam:.2f})', fontsize=16)

        for i in range(num_mix_pair):
            # Plot original image A
            axes[0, i].imshow(x_np[i])
            axes[0, i].set_title(f'Original Image A\nLabel: {y_a[i].item()}')
            axes[0, i].axis('off')

            # Plot original image B (the one mixed with A)
            # index = torch.where(y == y_b[i])[0][0].item()  # Find the index of y_b in the batch
            axes[1, i].imshow(x_np[index[i]])
            axes[1, i].set_title(f'Original Image B\nLabel: {y_b[i].item()}')
            axes[1, i].axis('off')

            # Plot mixed image
            axes[2, i].imshow(mixed_x_np[i])
            axes[2, i].set_title(f'Mixed Image\nLabel A: {y_a[i].item()} Label B: {y_b[i].item()}')
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.savefig('/cs/student/projects3/aibh/2023/jingqzhu/plots/visualise_mixup.jpg')