import torch
import torch.nn.functional as F  #激活函数    
import torch.utils.data as Data
import numpy as np

#该函数为模型主体，用于训练模型，并预测每一种语言的每一个类型学特征的结果
def run(target_language, feature_index, train_epoch, args = None) :
    torch.manual_seed(1) 

    #导入训练集数据
    x_train = torch.ones((1,args.input_size)).to(args.device)
    y_train = torch.ones(1).to(args.device)

    feature = args.features[feature_index]
    for l in args.languages:                                                #循环遍历各语言
        if l not in [target_language] and feature in list(args.INFO[l]):   #若该语言存在这一类型学特征，将其拼接至x_train
            x0 = args.all_embeddings[l]
            value = args.INFO[l][feature]
            y0 = torch.ones(10000) *args.features2num[feature][value]
            x_train = torch.cat((x_train, x0.to(args.device)), 0)
            y_train = torch.cat((y_train, y0.to(args.device)), )
                
    x_train = x_train[1:].type(torch.FloatTensor)
    y_train = y_train[1:].type(torch.LongTensor)     

    # 把 dataset 放入 DataLoader
    torch_dataset = Data.TensorDataset(x_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,              # 多线程数，windows只能写0
    )

    #导入测试数据
    x_test = args.all_embeddings[target_language].to(args.device)
    y_test = np.ones(10000)*args.features2num[feature][value]

    #搭建网络:mlp→relu→mlp
    output_size =  len(args.features2num[feature].keys())
    net = torch.nn.Sequential(
        torch.nn.Linear(args.input_size, args.hidden_dim),
        torch.nn.Dropout(args.hidden_dropout_prob),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_dim, output_size),
    )
    net.to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)     #优化器
    loss_func = torch.nn.CrossEntropyLoss()
    
    #训练
    for epoch in range(train_epoch):
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            optimizer.zero_grad() 
            out = net(batch_x) 
            loss = loss_func(out, batch_y) 
            loss.backward()  
            optimizer.step()

    #测试
    net.eval()
    out = net(x_test)
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.detach().cpu().numpy().squeeze()
    
    #print("accuracy:", np.sum(pred_y==y_test)/10000)
    return np.sum(pred_y==y_test)/10000                 #计算正确率

