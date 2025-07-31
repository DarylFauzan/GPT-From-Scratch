# Create datasets
import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class PretrainedDataset(Dataset):
    def __init__(self, tokens: list, block_size = 512, device = "cpu"):
        super().__init__()
        # fetch the text and label
        self.x = []
        self.y = []
        self.device = device
        self.block_size = block_size

        for i in range(len(tokens) - (block_size + 1)):
            self.x.append(tokens[i: i + block_size])
            self.y.append(tokens[i + 1: i + block_size + 1])

    def __len__(self): return len(self.x)

    def __getitem__(self, ix): 
        # convert them into pytorch's tensor
        x = torch.LongTensor(self.x[ix]).to(self.device)
        y = torch.Tensor([self.y[ix]]).squeeze().to(self.device)

        return x, y