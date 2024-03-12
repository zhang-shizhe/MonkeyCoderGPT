# MonkeyCoderGPT

# Overview
This project crafts **a character-level language model using a GPT-like architecture from scratch**. To showcase the model, a Flask-based deployment has been integrated, allowing for interactive demonstration.

![](https://github.com/zhang-shizhe/MonkeyCoderGPT/blob/main/img/web_page.gif)

The model is designed to understand and generate pythonic text one character at a time, allowing for a nuanced understanding of language syntax and structure. Given a brief starting prompt such as `import torch`, the model is capable of autonomously continuing to generate *pythonic* code. While the Python scripts produced by the model are syntactically accurate and contextually pertinent, they may not necessarily constitute functional or logically consistent programs. Thus, I decided to call it `MonkeyCoderGPT`.

## Features
- **Training on your own gaming laptop**: Total 10 Million of parameters, capable of training on consumer-level GPUs

- **Easy Deployment**: Deploy the model by running a single line of code.
- **Character-Level Tokenization**: Processes text data at the character level, capturing fine-grained linguistic patterns.
- **GPT-Like Architecture**: Utilizes a simplified version of the GPT architecture, including multi-head self-attention and feed-forward layers.
- **Customizable Model Configuration**: Allows easy adjustments of model parameters like the number of heads, head size, number of layers, and dropout rate.

## Corpus
I inclued and preprocessed some python scripts from the following open-source repositories to form my training and val data:
1. https://github.com/huggingface/transformers/tree/main
2. https://github.com/IBM/pytorch-seq2seq
3. https://github.com/allenai/allennlp
4. https://github.com/codertimo/BERT-pytorch
5. https://github.com/pytorch/examples

## Installation
Clone this repository to your local machine.
```bash
git clone git@github.com:zhang-shizhe/MonkeyCoderGPT.git
cd MonkeyCoderGPT
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Deployment
- Open a terminal in your project root directory.
- Run the application: **`python run.py`**.
- Open a web browser and go to http://127.0.0.1:5000/. You should see your landing page now.


## Development
### Generate Scripts without Training
To generate scripts with a pretained model, from root run:
```bash
cd src/monkeycodergpt/
python generate.py
```
This will load the state_dict uder models/.
Performance of the pretrained model: loss 0.68 on the val set after 5000 iterations. 

### Train your own MonkeyCoder with other corpus
Change `source_directories` inside build_corpus.py to include your own corpus or scripts.
Then from the root run:
```bash
cd src/monkeycodergpt/
python build_corpus.py
```

Modify the GPTConfig class in model.py to change the model configuration. Available parameters include:

- `vocab_size`: The number of unique characters in the dataset.
- `context_length`: The maximum length of the sequences the model during training can handle.
- `num_heads`: The number of attention heads in each multi-head attention layer.
- `head_size`: The dimensionality of each attention head.
- `emb_dim`: The dimensionality of the token embeddings.
- `num_layers`: The number of layers in the model.
- `dropout`: The dropout rate.

Train the model:
```bash
python train.py
```
My laptop has a RTX4070 GPU, and it takes about 30 minutes to train the model with default settings.
The state_dict will be stored under models/.

## Contributing
Welcome contributions to this project! Please feel free to submit issues or pull requests.
