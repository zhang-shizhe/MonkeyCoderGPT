import torch
from model import GPTLanguageModel, GPTConfig

from tqdm.auto import tqdm

batch_size = 64
context_length = 256
max_iters = 1000
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval = 100
eval_iters = 200

data_file = "sample_scripts.txt"

print(f"current device: {device}")

# read it in to inspect it
with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"unique chars: {''.join(chars)}")


# create a mapping from characters to integers
ch_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_ch = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    ch_to_idx[ch] for ch in s
]  # encoder: take a string, output a list of mapping idx
decode = lambda l: "".join(
    [idx_to_ch[idx] for idx in l]
)  # decoder: take a list of index, output a string

# encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split data to trainging and validation set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loader
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    start_idxs = torch.randint(len(data) - context_length, (batch_size,))
    context_idxs = torch.stack(
        [data[start_idx : start_idx + context_length] for start_idx in start_idxs]
    )
    target_idxs = torch.stack(
        [
            data[start_idx + 1 : start_idx + context_length + 1]
            for start_idx in start_idxs
        ]
    )

    context_idxs, target_idxs = context_idxs.to(device), target_idxs.to(device)

    return context_idxs, target_idxs


# init model and config

gpt_config = GPTConfig()
model = GPTLanguageModel(gpt_config).to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# compute loss on training and validation
@torch.no_grad()
def estimate_loss():
    res = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        res[split] = losses.mean()
    model.train()
    return res


# training process
for iter in tqdm(range(max_iters)):  # increase number of steps for good results...
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    context_idxs, target_idxs = get_batch("train")

    # evaluate the loss
    logits, loss = model(context_idxs, target_idxs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate texts
test_string = """import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

class Trainer:
"""
generated_text = decode(model.generate(
    context_idxs = torch.tensor(encode(test_string)).view(1, len(test_string)).to(device), 
    max_new_tokens=1000)[0].tolist()
               )

print(generated_text)

# save the output
output_filename = 'monkey_script.txt'
with open(output_filename, 'w', encoding='utf-8') as file:
    file.write(generated_text)
print(f"Generated text has been saved to {output_filename}")
