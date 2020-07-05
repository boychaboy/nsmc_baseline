import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

def read_data(train_path, test_path):
    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')
    return train, test

def get_dataloader(train, tokenizer, split=False, ratio=0.1, batch_size=32):
    # if split=True : split train / val dataset by ratio

    sentences = train['document']
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    labels = train['label'].values
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128
    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    # 어텐션 마스크 초기화
    attention_masks = []
    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    if split: #if train/val dataset
        # 훈련셋과 검증셋으로 분리
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2020, test_size=ratio)
        # 어텐션 마스크를 훈련셋과 검증셋으로 분리
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2020, test_size=ratio)

        # 데이터를 파이토치의 텐서로 변환
        train_inputs = torch.tensor(train_inputs)
        train_labels = torch.tensor(train_labels)
        train_masks = torch.tensor(train_masks)
        validation_inputs = torch.tensor(validation_inputs)
        validation_labels = torch.tensor(validation_labels)
        validation_masks = torch.tensor(validation_masks)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data) #val은 랜덤으로 뽑을 이유가 없으므로
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        return train_dataloader, validation_dataloader
    
    else:
        # 데이터를 파이토치의 텐서로 변환
        test_inputs = torch.tensor(input_ids)
        test_labels = torch.tensor(labels)
        test_masks = torch.tensor(attention_masks)

        # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
        # 학습시 배치 사이즈 만큼 데이터를 가져옴
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        
        return test_dataloader


