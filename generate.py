import torch
from model import GPTLanguageModel, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
data_file = "sample_scripts.txt"
print(f"current device: {device}")

# read it in to inspect it
with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
ch_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_ch = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    ch_to_idx[ch] for ch in s
]  # encoder: take a string, output a list of mapping idx
decode = lambda l: "".join(
    [idx_to_ch[idx] for idx in l]
)  # decoder: take a list of index, output a string


# Initialize the model and optimizer
gpt_config = GPTConfig(vocab_size=vocab_size)
model = GPTLanguageModel(gpt_config).to(device)

# Load the state dict
state_dict = torch.load('monkey_coder_gpt_state_dict.pt')
model.load_state_dict(state_dict)
model.eval()  # set the model to evaluation mode

# generate texts
test_string = """import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

class Trainer:
"""
generated_text = decode(model.generate(
    context_idxs=torch.tensor(encode(test_string)).view(1, len(test_string)).to(device), 
    max_new_tokens=10000)[0].tolist()
               )

# print(generated_text)

# save the output
output_filename = 'monkey_script.txt'
with open(output_filename, 'w', encoding='utf-8') as file:
    file.write(generated_text)
print(f"Generated text has been saved to {output_filename}")
