import argparse
from utils.train import train
from utils.load_data import get_iter
from utils.config import Config
from model.textcnn import TextCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,help='学习率 默认值为0.001')
parser.add_argument('--batch_size', type=int, default=128,help='每个batch中所含样本数 默认128')
parser.add_argument('--epoch', type=int, default=100,help='训练迭代的次数 默认10')
parser.add_argument('--filter_num', type=int, default=16,help='卷积核数量 默认16')
parser.add_argument('--filter_sizes', type=str, default='3,4,5',help='卷积核种类数 默认3,4,5')
parser.add_argument('--dropout', type=float, default=0.5,help='dropout率 默认0.5')
args = parser.parse_args()

config = Config()

UTTER,LABEL,train_iter,dev_iter = get_iter(config,args)
vocab_size = len(UTTER.vocab)                  # 词表大小
class_num = 2                 # 类别数目
embedding_dim = UTTER.vocab.vectors.size()[-1] # 词向量维度
vectors = UTTER.vocab.vectors                  # 词向量
print(vocab_size)
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

textcnn_model = TextCNN(
	class_num=class_num,
	filter_sizes=args.filter_sizes,
	filter_num=args.filter_num,
	vocabulary_size=vocab_size,
	embedding_dimension=embedding_dim,
	vectors=vectors,
	dropout=args.dropout
	)

try:
    train(train_iter,dev_iter,textcnn_model,args,config)
except KeyboardInterrupt:
	print("提前停止")