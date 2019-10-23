import argparse
from seq2seq2seq import seq2seq2seq
import tensorflow as tf
from tensorflow.python import debug as tf_debug

def parse():
    parser = argparse.ArgumentParser(description="variational autoencoder")
    parser.add_argument('-model_dir', default='train_model',help='output model weight dir')
    parser.add_argument('-model_path', help='latest model path')
    parser.add_argument('-batch_size', default=96,type=int,help='batch size')
    parser.add_argument('-latent_dim', default=300,type=int,help='laten size')
    parser.add_argument('-data_dir', default='chinese_data',help='data dir')
    parser.add_argument('-saving_step', default=1000,type=int,help='saving step')
    parser.add_argument('-num_steps',default=20000,type=int,help='number of steps')
    parser.add_argument('-source_length',default=50,type=int,help='source sentence length')
    parser.add_argument('-code_length',default=13,type=int,help='code sentence length')
    parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-train',action='store_true',help='whether train')
    parser.add_argument('-pretrain',action='store_true',help='whether pretrain')
    parser.add_argument('-test',action='store_true',help='whether test')
    parser.add_argument('-test_input', default='giga_word/test/input.txt',help='path of testing input')
    parser.add_argument('-test_output', default='giga_word/test/result.txt',help='path of result file')
    parser.add_argument('-pretrain_input', default='giga_word/pretrain_article.txt', help='input path for pretraining generator')
    parser.add_argument('-pretrain_target', default='giga_word/pretrain_target.txt', help='target path for pretraining generator')
    parser.add_argument('-summary', default='giga_word/train.title.txt', help='summary path as real data for discriminator')
    parser.add_argument('-article', default='giga_word/train.article.txt', help='article path for unparalleled training')
    args = parser.parse_args()
    return args

def run(args):
    sess = tf.Session()
    model = seq2seq2seq(args,sess)
    if args.train:
        model.train()
    if args.pretrain:
        model.pretrain()
    if args.test:
        model.test()

if __name__ == '__main__':
    args = parse()
    run(args)