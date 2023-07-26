import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict
import json

from model import PoemGPT


device = "cuda" if torch.cuda.is_available() else "cpu"

with open("./config/poem_gpt_config.json", "r") as f:
    config = json.load(f)

with open("./data/input.txt", "r", encoding="utf-8") as f:
    data = f.read()


chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(data))

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_random_batch(split: str="train") -> Tuple[torch.Tensor, torch.Tensor]:

    data = train_data if split=="train" else val_data

    batch_size = config["batch_size"]
    block_size = config["block_size"]

    idxs = torch.randint(0, len(data)-block_size, size=(batch_size,))
    x_batch = torch.stack([data[i:i+block_size] for i in idxs])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in idxs])

    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    return x_batch, y_batch


@torch.no_grad()
def eval_model() -> Dict[str, float]:
    losses = {}
    poem_gpt.eval()

    for split in ["train", "val"]:
        data = train_data if split=="train" else val_data
        loss = 0
        for iter in range(config["eval_iters"]):
            x_batch, y_batch = get_random_batch(split)
            _, l_ = poem_gpt(x_batch, y_batch)
            loss += l_.item()
        
        losses[split] = loss/config["eval_iters"]
    
    poem_gpt.train()
    return losses


def train_poem_gpt():

    for iter in range(config["train_iters"]):

        if iter%config["eval_interval"]==0:
            losses = eval_model()
            print(f"iter {iter} train_loss: {losses['train']} val_loss: {losses['val']}")
        
        x_batch, y_batch = get_random_batch()
        _, loss = poem_gpt(x_batch, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


poem_gpt = PoemGPT(config, vocab_size)
poem_gpt = poem_gpt.to(device)
optimizer = torch.optim.AdamW(params=poem_gpt.parameters(),
                              lr=config["learning_rate"])

train_poem_gpt()
