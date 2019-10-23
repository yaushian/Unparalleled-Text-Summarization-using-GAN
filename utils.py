from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import numpy as np
import random
import json
import os
import re
import sys

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def write_json(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

class utils():    
    def __init__(self,args):
        self.source_length = args.source_length
        self.code_length = args.code_length
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.pre_input = args.pretrain_input
        self.pre_target = args.pretrain_target
        self.summary_path = args.summary
        self.article_path = args.article

        self.word_id_dict = read_json('giga_word/new_word_id_dict.json')
        self.vocab_size = len(self.word_id_dict)

        self.BOS_id = self.word_id_dict['__BOS__']
        self.EOS_id = self.word_id_dict['__EOS__']
        self.UNK_id = self.word_id_dict['<unk>']
        
        self.index2word = [[]]*len(self.word_id_dict)
        for word in self.word_id_dict:
            self.index2word[self.word_id_dict[word]] = word



    def sent2id(self,sent,length,max_unknown=None):
        sent = sent.strip()
        vec = np.zeros((length),dtype=np.int32)

        i = 0
        unknown = 0
        for word in sent.strip().split():
            if word in self.word_id_dict and word != '<unk>':
                vec[i] = self.word_id_dict[word]
                i += 1
            else:
                unknown += 1
            
            if i>=length:
                break
        if max_unknown is None:
            return vec
        else:
            return vec,unknown,i



    def id2sent(self,indices):
        sent = []
        QQ = dict()
        for index in indices:
            if index in QQ:
                continue
            QQ[index] = 1
            if index <= 1:
                break
            sent.append(self.index2word[index])
        return ' '.join(sent)


    def file_data_generator(self,filename,shuffle=False):
        while(1):
            with open(filename) as fp:
                for line in fp:
                    if shuffle:
                        if random.randint(0,9)>=5:
                            yield line.strip()
                    else:
                        yield line.strip()
    

    def gan_data_generator(self):
        one_r_batch = [];one_s_batch = [];one_w_batch = []
        for real_sentence,source_sentence in zip(self.file_data_generator(self.summary_path,True),\
                                                 self.file_data_generator(self.article_path,True)):
            r_vec,r_unk,r_len = self.sent2id(real_sentence,self.code_length, max_unknown=0)
            s_vec,s_unk,s_len = self.sent2id(source_sentence,self.source_length, max_unknown=2)
            one_r_batch.append(r_vec)
            one_s_batch.append(s_vec)

            if len(one_r_batch)==self.batch_size:
                random.shuffle(one_r_batch)
                random.shuffle(one_s_batch)
                yield np.array(one_s_batch),np.array(one_r_batch)
                one_r_batch = []
                one_s_batch = []


    def pretrain_generator_data_generator(self):
        one_x_batch = [];one_y_batch = []
        for source_sentence,title_sentence in zip(self.file_data_generator(self.pre_input),self.file_data_generator(self.pre_target)):
            x_vec,x_unk,x_len = self.sent2id(source_sentence,self.source_length, max_unknown=0)
            y_vec,y_unk,y_len = self.sent2id(title_sentence,self.code_length, max_unknown=2)
            
            one_x_batch.append(x_vec)
            one_y_batch.append(y_vec)
            if len(one_x_batch)==self.batch_size:
                yield np.array(one_x_batch),np.array(one_y_batch)
                one_x_batch = [];one_y_batch = []


    def test_data_generator(self, input_path):
        one_x_batch = []
        for source_sentence in self.file_data_generator(input_path):
            one_x_batch.append(self.sent2id(source_sentence,self.source_length))
            if len(one_x_batch)==self.batch_size:
                yield np.array(one_x_batch)
                one_x_batch = []