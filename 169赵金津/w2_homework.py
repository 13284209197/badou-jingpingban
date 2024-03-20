import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
class mymodel(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 10)
        self.linear2 = nn.Linear(10, 20)
        self.linear3 = nn.Linear(20, 5)
        #不能再经过softmax，会导致映射关系发生变化，不能正确计算loss
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x_l1 = self.linear1(x)
        x_l2 = self.linear2(x_l1)
        y_pred = self.linear3(x_l2)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def get_data(datanum, classnum):
    Y = []
    X = []
    for i in range(datanum):
        x = np.random.random(classnum)
        y = np.argmax(x)
        X.append(x)
        Y.append(int(y))
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    X,Y = get_data(test_sample_num, 5)
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model.forward(X)
        for y_p,y_t in zip(y_pred,Y):
            n = np.argmax(y_p)
            if n==y_t:
                correct += 1
            else:
                wrong +=1
    print("预测正确个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct/test_sample_num




def main():
    #parameter
    train_num = 1000
    batchsize = 50
    class_num = 5
    learning_rate = 0.001
    Epochs = 1000
    log = []
    print("===============加载数据集================")
    X_train,y_train = get_data(train_num,class_num)
    print("===============搭建网络================")
    Model = mymodel(class_num)
    #优化器
    optim = torch.optim.Adam(Model.parameters(), lr= learning_rate)
    for i in range(Epochs):
        Model.train()
        watch_loss = []
        for batchindex in range(train_num//batchsize):
            trainx = X_train[batchindex*batchsize: (batchindex+1)*batchsize]
            trainy = y_train[batchindex * batchsize: (batchindex + 1) * batchsize]
            #计算损失
            loss = Model.forward(trainx,trainy)
            #计算梯度
            loss.backward()
            #更新梯度
            optim.step()
            #重置梯度
            optim.zero_grad()
            #取出张量
            watch_loss.append(loss.item())
        print("=======\n第%d轮训练平均loss值为%f" % (i+1, np.mean(watch_loss)))
        acc = evaluate(Model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(Model.state_dict(),"homework.pt")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()













if __name__ == "__main__":
    main()