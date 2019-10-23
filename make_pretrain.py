import random
import numpy as np
import sys
import os
import re
import json
import struct
import csv
from tensorflow.core.example import example_pb2
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def swap(l,id1,id2):
    e1 = l[id1]
    l[id1] = l[id2]
    l[id2] = e1
    return l

class data_maker():
    def __init__(self):
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        self.word_id_dict = read_json('giga_word/new_word_id_dict.json')


    def make_pretrain_data(self):
        result = []
        source_fp = open('giga_word/pretrain_article.txt','w')
        target_fp = open('giga_word/pretrain_target.txt','w')
        for line in open('giga_word/train.article.txt'):
            word_list = line.strip().split()
            length = len(word_list)
            if length <= 20 or random.randint(0,1) == 1:
                continue
            random_length = random.randint(6,11)
            random_start = random.randint(0,10)

            target = []
            source = []
            phrase = []
            for j,word in enumerate(word_list):
                if word in self.word_id_dict and word != ',' and word != '<unk>':
                    if len(target) < random_length and j >= random_start:
                        target.append(word)
                if random.randint(0,15) == 1:
                    continue
                phrase.append(word)
                if random.randint(0,2) >= 2:
                    source.append(' '.join(phrase))
                    phrase = []
            if len(phrase) > 0:
                source.append(' '.join(phrase))
            random.shuffle(source)
            result.append([' '.join(source),' '.join(target)])

        random.shuffle(result)
        for r in result:
            source_fp.write(r[0] + '\n')
            target_fp.write(r[1] + '\n')


if __name__=='__main__':
    maker = data_maker()
    maker.make_pretrain_data()