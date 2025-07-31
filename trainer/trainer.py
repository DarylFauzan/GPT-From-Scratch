import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class TrainConfig():
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    checkpoints: int = 10 # the model every <checkpoints> epochs

class PretrainedTrainer:
    def __init__(self, model, training_config, train_dataset, val_dataset, task = "binary", device = "cuda"):
        """This is the training function. Here are some of the parameters:
    
        1. model: your Pytorch model.
        2. train_dataset: Your dataset for training (pytorch.utils.data.Dataset).
        3. val_dataset: Your dataset for validation (pytorch.utils.data.Dataset).
        4. training_config: Fill the training config with the TrainConfig class.
        """
        assert task in {"binary", "multiclass"}, "task only accept binary or multiclass"
        # define the training config
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = training_config.epochs
        self.batch_size = training_config.batch_size
        self.num_workers = training_config.num_workers
        self.checkpoints = training_config.checkpoints
        self.device = device

        # define the criterion
        if task == "binary":
            self.criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)
        # define the optimizers
        self.optimizer = AdamW(model.parameters(), 
                               lr = training_config.learning_rate,
                               weight_decay = training_config.weight_decay)

    def create_report_plot(self, loss_train, loss_val, f1_train, f1_val):
        x = [i for i in range(1, self.epochs + 1)]

        metrics = {"Train_Loss": loss_train,
               "Val_Loss": loss_val,
               "Train_F1Score": f1_train,
               "Val_F1Score": f1_val}

        # Create plot for each metric
        for key in metrics.keys():
            fig, ax = plt.subplots(figsize = (10, 10))

            sns.lineplot(x = x, y = metrics[key])
            plt.title(f"{key}")
            plt.xlabel("epochs")
            plt.ylabel(f"key")

            fig.savefig(f"model/reports/plots/{key}.png", dpi = 600)
    

    def train(self):
        # create dataloader
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = True, 
                                  num_workers = self.num_workers,
                                  pin_memory = True if self.device == "cuda" else False)
        
        val_loader = DataLoader(self.val_dataset, 
                                batch_size = self.batch_size, 
                                num_workers = self.num_workers,
                                pin_memory = True if self.device == "cuda" else False)        
        
        loss_train = []
        loss_val = []

        f1_train = []
        f1_val = []
        print(f"""
 ||  ||  
 \\()// 
//(__)\\
||    ||
Trainable Parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad):d} / {sum(p.numel() for p in self.model.parameters())}
Start Training 🧠...""")
        with open("model/reports/log.txt", "w") as f:
            for epoch in range(self.epochs):
                epoch_train_loss, epoch_val_loss = 0, 0
                # Add tqdm wrapper
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                    # train the model
                    self.model.train()
                    for x, y in pbar:
                        self.optimizer.zero_grad()
                        # create prediction and compute loss
                        _y, loss = self.model(x, y)
                        loss.backward()
                        self.optimizer.step()
    
                        # add the training loss
                        epoch_train_loss += loss.item()
                        pbar.set_postfix(train_loss=loss.item())
    
                # compute the validation set
                self.model.eval()
                for x, y in val_loader:
                    _y = self.model(x) 
                    loss = self.criterion(_y, y)
    
                    # add the validation loss
                    epoch_val_loss += loss.item()
    
                # compute the average of loss and metrics in one epoch
                avg_epoch_train_loss = epoch_train_loss / len(train_loader)
                avg_epoch_val_loss = epoch_val_loss / len(val_loader)
    
                # append it to the log
                loss_train.append(avg_epoch_train_loss)
                loss_val.append(avg_epoch_val_loss)
    
                # Display validation loss after tqdm loop ends
                summary = f"Epoch {epoch+1}/{self.epochs}: train_loss = {avg_epoch_train_loss:4f}, val_loss = {avg_epoch_val_loss:4f}"
                print(summary)

                f.write(summary)

                # checkpoints
                if epoch % self.checkpoints == 0:
                    torch.save(self.model.state_dict(), 
                               f"model/reports/checkpoints/_{epoch//self.checkpoints}_Jerro.pth") 
        
        # Create plot after training
        self.create_report_plot(loss_train, loss_val, f1_train, f1_val)

        # Save the final model
        torch.save(self.model.state_dict(), 
                    f"model/JerroSentimentAnalysis.pth") 