import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import GPT2Tokenizer
from model import JeroGPT, ModelConfig

import torch
from trainer.datasets import PretrainedDataset
from trainer.trainer import PretrainedTrainer, TrainConfig

#device
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# get all the articles
articles = os.listdir("trainer/data")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)

# store all the articles in a chunk
chunks_data = []
for article in articles:
    loader = PyMuPDFLoader(f"trainer/data/{article}")
    documents = loader.load()
    split = text_splitter.split_documents(documents)
    chunk_texts = [chunk.page_content for chunk in split]
    chunk_texts[-1] += "<|endoftext|>"
    chunks_data.extend(chunk_texts)

# tokenize the chunk text
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer/local_gpt2_tokenizer")
chunk_tokens = list(map(tokenizer.encode, chunks_data))

# store it all in a single list
data = []
for c in chunk_tokens:
    data.extend(c)

# create train and val data
train_size =int(len(data) * 0.8)
train_data = data[:train_size]
val_data = data[train_size:]

train_dataset = PretrainedDataset(train_data, device = device)
val_dataset = PretrainedDataset(val_data, device = device)

# define the model
model_config = ModelConfig()
model = JeroGPT(model_config).to(device)

# define the trainer
trainer_config = TrainConfig()
trainer = PretrainedTrainer(model, trainer_config, train_dataset, val_dataset, task = "binary", device = device)

trainer.train()