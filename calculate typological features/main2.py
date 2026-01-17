# coding=gbk#coding:utf-8
import openpyxl
from openpyxl import Workbook
import argparse
import os
import pickle
import torch
import copy
from calculate import run
from statistics import write


#主函数
def main(args):
    #merge embeddings：将各种语言的10000条例句的向量表示纵向拼接到一起
    
    path = './'+args.model+'/layer '+str(args.layer)
    all_embeddings = dict()
    for l in args.languages:
        filename = path+'/'+l+'_'+args.model+'_embedding.dat'
        with open(filename, 'rb') as fp:
            value = torch.tensor(pickle.load(fp))
            all_embeddings[l] = value
    args.all_embeddings = all_embeddings
    

    #计算每种语言的每一类型学特征的预测结果，并写入txt文件中
    for target_language in languages:                                            #循环语种
        for feature_index in range(len(args.features)):                          #循环特征
            feature = args.features[feature_index]
            if feature in list(args.INFO[target_language].keys()):
                for train_epoch in range(1, args.num_train_epochs+1):            #循环训练轮数
                    acc = run(target_language, feature_index, train_epoch, args)
                    with open(args.filename, 'a', encoding='utf-8') as fp:       #将结果写入到目标文件中
                        fp.write("language:"+str(target_language)+" \t"
                                 +"feature_index:"+str(feature_index)+"\t"
                                 +"train_epoch:"+str(train_epoch)+"\t"
                                 +"acc:"+str(acc)+'\n')


if __name__ == "__main__":
    #重要的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'xlm', 'xlm-R'])    #选择哪个预训练多语言模型
    parser.add_argument('--num_train_epochs', type=int, default=5)                                #训练轮数
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)                         #隐层神经元的dropout率
    parser.add_argument('--learning_rate', type=float, default=1e-2)                              #学习率
    parser.add_argument('--input_size', type=int, default=768, choices=[768, 1024])               #输入隐层时的向量维度
    parser.add_argument('--hidden_dim', type=int, default=100)                                    #隐层神经元个数
    parser.add_argument('--train_batch_size', type=int, default=512)                              #训练时的batch_size大小
    parser.add_argument('--layer', type=int, default=1)                                           #对模型哪一隐层进行训练

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                 #设备(cpu or gpu)
    args.filename = 'result_'+args.model+'_'+'layer'+str(args.layer)+'.txt'                      #结果的文件名

    print(args)

    #把数值化后的类型学特征信息导入进来
    with open('features2num_WALS.dat', 'rb') as fp:
        features2num_WALS = pickle.load(fp)
    with open('features2num_SSWL.dat', 'rb') as fp:
        features2num_SSWL = pickle.load(fp)
    args.features2num = {**features2num_WALS, **features2num_SSWL}   #合并两个字典

    #把整理的各种语言的类型学信息导入进来
    with open('WALS_INFO_dict.dat', 'rb') as fp:
        WALS_INFO = pickle.load(fp)
    with open('SSWL_INFO_dict.dat', 'rb') as fp:
        SSWL_INFO = pickle.load(fp)
    args.INFO = copy.deepcopy(WALS_INFO)      #合并两个字典
    for l in SSWL_INFO.keys():
        for f in  SSWL_INFO[l].keys():
            args.INFO[l][f] = SSWL_INFO[l][f]


    args.features = list(features2num_WALS.keys()) + list(features2num_SSWL.keys())     #类型学特征变量
    languages = list(WALS_INFO.keys())                                                  #语种变量
    args.languages = languages

    main(args)
    write(args)
