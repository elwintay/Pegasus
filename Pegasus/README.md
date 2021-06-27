# Pegasus Abstractive Text Summarisation
With reference to: [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777v2.pdf)

> This project is about implementing and finetuning the Pegasus model using Pytorch and Transformers library. Dataset used in this project is the BBC news Dataset. It also provides instructions on how to implement the model offline. This is just an experimental project. Feel free to change the config, or make your own config file and argparse it in.

## Package Description
```
Pegasus/
├─ Src/
    ├── data_preprocessing.py: text prepcessing functions
    ├── prepare_dataset.py: convert text to pytorch tensors
├─ Data/: raw training data of text and summaries
    ├── Summary/ :folder for all the summaries of the texts
    ├── Text/ :folder for all the texts
├─ Pretrained/: save pre-trained Pegasus models
    ├── model/:
        ├── pegasus-original/: contains config.json and pytorch_model.bin of the original pegasus model
            ├── config.json
            ├── pytorch_model.bin
        ├── pegasus-finetuned/: contains config.json and pytorch_model.bin of the finetuned pegasus model
    ├── tokenizer/ : Saved Pegasus tokeniser
        ├── pegasus-tokenizer/ : contains special_tokens_map.json, spiece.model and tokenizer_config.json of pegasus model
            ├── special_tokens_map.json
            ├── spiece.model
            ├── tokenizer_config.json
├─ Text Input/: Text and summary (if available) you would like to summarise
    ├── Summary/ :folder for all the summaries of the texts
    ├── Text/ :folder for all the texts
├─ Notebooks/:
    ├─ Pegasus.ipynb: Example pegasus summarisation notebook
├─ requirements.txt: Pypi packages to install and their versions
├─ Dockerfile: Setup to run on docker container
├─ finetune.py: To finetune the pegasus model
├─ pegasus.py: To generate summaries
```

## Environments

- python         (3.6)
- cuda           (10.2)
- Ubuntu-18.04.5  

## Instructions
- To setup container
```bash
>>  sudo docker-compose run --service-ports pegasus 
```
- To finetune, within the container
```bash
>> python finetune.py
```
- To generate summaries
```bash
>> python pegasus.py
```
