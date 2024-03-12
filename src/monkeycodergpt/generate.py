import torch
import os
from model import GPTLanguageModel, GPTConfig


def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data_file = "../../data/sample_scripts.txt"
    # model_file = '../../models/monkey_coder_gpt_state_dict.pt'
    # print(f"current device: {device}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '../../data/sample_scripts.txt')
    model_file = os.path.join(current_dir, '../../models/monkey_coder_gpt_state_dict.pt')

    # read it in to inspect it
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    ch_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_ch = {i: ch for i, ch in enumerate(chars)}

    # encoder: take a string, output a list of mapping idx
    def encode(s): return [ch_to_idx[ch] for ch in s]
    # decoder: take a list of index, output a string
    def decode(l): return "".join([idx_to_ch[idx] for idx in l])  

    # Initialize the model and optimizer
    gpt_config = GPTConfig(vocab_size=vocab_size)
    model = GPTLanguageModel(gpt_config).to(device)

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # generate texts
    test_string = """import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

class Trainer:
    """
    context_idxs = torch.tensor(encode(test_string)).view(1, len(test_string)).to(device)
    generated_idxs = model.generate(context_idxs, max_new_tokens=300).tolist()[0]
    generated_text = decode(generated_idxs)

    return generated_text

    # # save the output
    # output_filename = '../../data/monkey_script.txt'
    # with open(output_filename, 'w', encoding='utf-8') as file:
    #     file.write(generated_text)
    # print(f"Generated text has been saved to {output_filename}")


# For testing purposes
if __name__ == "__main__":
    print(generate())