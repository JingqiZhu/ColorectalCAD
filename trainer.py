import time
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, device, selection_metric='mcc', num_epochs=10, mixup_option='none', mixup_alpha=0, data_loader_factory=None):
        """
        Initialize the Trainer class with all necessary components.

        Args:
            model (nn.Module): The model to be trained.
            dataloaders (dict): Dictionary containing training, validation, and test data loaders.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            device (torch.device): The device to run the model on (CPU or GPU).
            selection_metric (str): The metric to select the best model ('f1', 'bacc', 'mcc').
            num_epochs (int): Number of epochs for training.
            mixup_option (str): Option for mixup ('none', 'mixup', 'balanced_mixup').
            data_loader_factory (object): Instance of DataLoaderFactory to access mixup methods.
        """
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.selection_metric = selection_metric
        self.num_epochs = num_epochs
        self.mixup_option = mixup_option
        self.mixup_alpha = mixup_alpha
        self.data_loader_factory = data_loader_factory

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def train(self):
        """
        Train the model and evaluate it on the validation set after each epoch.
        """
        since = time.time()

        dataset_sizes = {x: len(self.dataloaders[x].dataset) for x in self.dataloaders}

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                all_preds = []
                all_labels = []

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    if phase == 'train':
                        if self.mixup_option == 'mixup':
                            inputs, labels_a, labels_b, lam, _ = self.data_loader_factory.mixup_data(inputs, labels)
                            labels_a = labels_a.to(self.device)
                            labels_b = labels_b.to(self.device)
                        elif self.mixup_option == 'balanced_mixup':
                            inputs, labels_a, labels_b, lam, _ = self.data_loader_factory.balanced_mixup_data(inputs, labels)
                            labels_a = labels_a.to(self.device)
                            labels_b = labels_b.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train' and self.mixup_option in ['mixup', 'balanced_mixup']:
                            loss = self.data_loader_factory.mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                        else:
                            loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data) 

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                epoch_loss = running_loss / dataset_sizes[phase]

                if self.selection_metric == 'f1':
                    epoch_metric = f1_score(all_labels, all_preds, average='macro')
                    metric_name = 'Macro F1'
                elif self.selection_metric == 'bacc':
                    epoch_metric = balanced_accuracy_score(all_labels, all_preds)
                    metric_name = 'Balanced Accuracy'
                elif self.selection_metric == 'mcc':
                    epoch_metric = matthews_corrcoef(all_labels, all_preds)
                    metric_name = 'MCC'
                else:
                    raise ValueError("Unsupported selection criterion")

                print(f'{phase} Loss: {epoch_loss:.4f} {metric_name}: {epoch_metric:.4f}')

                if phase == 'train':
                    self.train_losses.append(epoch_loss)
                    self.train_metrics.append(epoch_metric)
                    self.scheduler.step()  # Step the scheduler at the end of the epoch
                else:
                    self.val_losses.append(epoch_loss)
                    self.val_metrics.append(epoch_metric)

                    if epoch_metric > self.best_metric:
                        self.best_metric = epoch_metric
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val {metric_name}: {self.best_metric:.4f}')

        self.model.load_state_dict(self.best_model_wts)

        self.save_model() # save model weights

        self._plot_metrics(metric_name)

        return self.model

    def evaluate(self, dataloader, class_names):
        """
        Evaluate the model on the test set.

        Args:
            dataloader (DataLoader): The DataLoader for the test dataset.
            class_names (list): List of class names.
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        example_losses = []

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        total_loss = running_loss / len(dataloader.dataset)
        f1 = f1_score(all_labels, all_preds, average='macro')
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        print(f'Test Loss: {total_loss:.3f} Macro F1: {f1:.3f} Balanced Accuracy: {balanced_acc:.3f} MCC: {mcc:.3f}')

        # Print classification report
        print("Classification Report:")
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
        print(report)

        # Compute and display confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig('/cs/student/projects3/aibh/2023/jingqzhu/output/confusion_matrix.png')
        plt.close()

        # Plot and save histogram of example losses
        plt.figure(figsize=(10, 5))
        plt.hist(example_losses, bins=50, alpha=0.75)
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.title('Histogram of Example Losses')
        plt.savefig('/cs/student/projects3/aibh/2023/jingqzhu/output/example_loss_histogram.png')
        plt.close()

    def _plot_metrics(self, metric_name):
        """
        Plot the training and validation metrics.

        Args:
            metric_name (str): Name of the metric used for evaluation.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.num_epochs), self.train_losses, label='Train Loss')
        plt.plot(range(self.num_epochs), self.val_losses, label='Validation Loss')
        plt.plot(range(self.num_epochs), self.train_metrics, label=f'Train {metric_name}')
        plt.plot(range(self.num_epochs), self.val_metrics, label=f'Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Metric')
        plt.title(f'Training and Validation Loss and {metric_name}')
        plt.legend()
        plt.savefig('/cs/student/projects3/aibh/2023/jingqzhu/output/plot_losses.png')
        plt.close()

    def save_model(self, file_name='best_model.pth'):
        """Saves the model weights to a specified file."""
        save_path = os.path.join('/cs/student/projects3/aibh/2023/jingqzhu/output', file_name)
        torch.save(self.best_model_wts, save_path)
        print(f'Model weights saved to {save_path}')

    
