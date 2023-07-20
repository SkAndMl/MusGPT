import torch
import json
import time
import argparse
from model import PoemGPT
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("./config/poem_gpt_config.json") as f:
    config = json.load(f)

with open("./data/input.txt", "r", encoding="utf=8") as f:
    data = f.read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    itos = {i: ch for i, ch in enumerate(chars)}

decode = lambda l: "".join([itos[i] for i in l]) 

poem_gpt = PoemGPT(config=config,
                   vocab_size=vocab_size)
poem_gpt.load_state_dict(torch.load("./weights/poem_gpt_weights.pt",
                                    map_location=torch.device(device=device)))



def print_like_gpt(text):

    for ch in text:
        print(ch, end="", flush=True,)
        time.sleep(0.02)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_tokens", required=True, metavar="", type=int)
    args = parser.parse_args()
    out = poem_gpt.generate(max_new_tokens=args.num_tokens) # [1, S]
    text = decode(out[0].cpu().numpy())[1:]
    print_like_gpt(text=text)
