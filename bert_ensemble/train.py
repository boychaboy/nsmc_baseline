__author__ = 'boychaboy'

import tensorflow as tf
import torch
import torch.nn as nn
    
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import argparse

from preprocess import get_dataloader, read_data

def train(model, train_dataloader, validation_dataloader, test_dataloader, epochs=10, optimizer='AdamW'):
    pass


def main():
    args_parser = argparse.ArgumentParser(description='Tuning BERT for Sentiment Analysis')
    args_parser.add_argument('--model', choices=['multilingual-bert','kobert','hanbert'], required=True)
    args_parser.add_argument('--train_path', required=True)
    args_parser.add_argument('--test_path', required=True)

    args_parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=32, help='Number of sentences in each mini-batch')
    args_parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    args_parser.add_argument('--split_ratio', type=float, default=0.1, help='Train/Validation set split ratio')
    
    
    





if __name__ == '__main__':
    main()
