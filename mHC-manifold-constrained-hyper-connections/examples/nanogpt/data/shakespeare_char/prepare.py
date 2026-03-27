import json
import os
import urllib.request

import torch

data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

data_dir = os.path.dirname(__file__)
input_path = os.path.join(data_dir, "input.txt")

if not os.path.exists(input_path):
    urllib.request.urlretrieve(data_url, input_path)

with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()

chars = sorted(list(set(data)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encoded = torch.tensor([stoi[ch] for ch in data], dtype=torch.long)

n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]

torch.save(train_data, os.path.join(data_dir, "train.bin"))
torch.save(val_data, os.path.join(data_dir, "val.bin"))

meta = {"vocab_size": len(chars), "itos": itos, "stoi": stoi}

with open(os.path.join(data_dir, "meta.json"), "w") as f:
    json.dump(meta, f)
