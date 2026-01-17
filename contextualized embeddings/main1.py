# coding=gbk#coding:utf-8
import argparse
import torch
from transformers import BertTokenizer, BertModel, XLMTokenizer, XLMModel,XLMRobertaModel, XLMRobertaTokenizer
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
    for layer in range(1,13):
        b = []
        for line in lines:
            if args.model == 'bert':
                sent = "[CLS] " + line.strip() + " [SEP]"
            else:
                sent = line.strip()
            tokenized_text = tokenizer.tokenize(sent)                                        #对句子做token化处理
            if len(tokenized_text) > args.max_len:                                           #截长补短
                tokenized_text = tokenized_text[:args.max_len]
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)                 #转化成数字id
            segments_ids = [1] * len(tokenized_text)                                         #确定分割点

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)  #得到句子的向量化表示
                encoded_layers = outputs[-1][layer]
                token_vecs = encoded_layers[0]
                #根据不同模型，采用不同的向量化表示方法
                if args.model == 'bert':
                    sentence_embedding = token_vecs[0]                                       #取[SEP]的向量表示
                else:
                    sentence_embedding = torch.mean(token_vecs, dim=0)                       #均值化处理
                b.append(sentence_embedding.detach().numpy())

        #存储结果
        path = './'+args.model+'/layer '+str(layer)
        if not os.path.exists(path):
            os.makedirs(path)
        newfile = path +'/'+file.split('.')[0]+'_'+args.model+'_embedding.dat'
        with open(newfile, 'wb') as f:
            pickle.dump(b, f)

#主函数
def main(args):
    #条件语句，用于分类判断具体是采用哪一个预训练多语言模型
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')                 #导入打token的工具
        model = BertModel.from_pretrained("bert-base-multilingual-cased")                         #导入预训练模型文件
    elif args.model == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")                       #xlm-mlm-xnli15-102
        model = XLMModel.from_pretrained("xlm-mlm-tlm-xnli15-1024")
    elif args.model == 'xlm-R':
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    model.eval()                              #模型评估
    files = os.listdir(args.source_text)
    for file in files:                       #运行文件夹下各个语种的文件
        run(file, tokenizer, model)


if __name__ == "__main__":
    #重要的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xlm', choices=['bert', 'xlm', 'xlm-R'])    #选择哪一个预训练多语言模型
    parser.add_argument('--max_len', type=int, default=512)                                      #句子最大长度
    parser.add_argument('--source_text', type=str, default='./multilingual text')                #语料的文件夹

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                #设备(gpu or cpu)

    print(args)
    main(args)
