# coding=gbk#coding:utf-8
#本文件随机mbert对句子的向量化表示
import argparse
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig
import os
import numpy as np
import pickle


#run函数，用于将各语言的句子进行向量化表示
def run(file, tokenizer, model):
    #导入文本
    with open(args.source_text+'/'+file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

    #按模型的各隐层进行循环
    b = []
    for line in lines:
        encoded_input = tokenizer(line.strip(), return_tensors='pt')
        outputs = model(**encoded_input)
        encoded_layers = outputs[0]
        token_vecs = encoded_layers[0]
        sentence_embedding = token_vecs[0]
    
        b.append(sentence_embedding.detach().numpy())    #得到句子的向量化表示

        #存储结果
        path = './'+args.model+'/layer '+str(12)
        if not os.path.exists(path):
            os.makedirs(path)
        newfile = path +'/'+file.split('.')[0]+'_'+args.model+'_embedding.dat'
        with open(newfile, 'wb') as f:
            pickle.dump(b, f)


#主函数
def main(args):
    configuration = BertConfig.from_pretrained('config.json')                       #导入配置
    model = BertModel(configuration)                                                #构建模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')       #打token的工具

    model.eval()                               #模型评估
    files = os.listdir(args.source_text)
    for file in files:                         #运行文件夹下各个语种的文件
        run(file, tokenizer, model)


if __name__ == "__main__":
    #重要的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert_random')                  #选择哪一个预训练多语言模型
    parser.add_argument('--max_len', type=int, default=512)                          #句子最大长度
    parser.add_argument('--source_text', type=str, default='./multilingual text')    #语料的文件夹

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    #设备(gpu or cpu)

    print(args)
    main(args)
