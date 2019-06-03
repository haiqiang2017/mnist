# _*_ coding:utf-8 _*_
import numpy as np
import random
import mnist_loader

class cnn1(object):
    def __init__(self,sizes):
        self.layers = len(sizes)
        self.weights = [np.random.randn(x,y) for (x,y) in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
    def SGD(self,train_data,epoches,mini_size,eta,test_data):
        if test_data: n_test = len(test_data)
        n = len(train_data) #表示 训练样本数量
        for epoch in range(epoches):
            random.shuffle(train_data)

            mini_batches = [train_data[k:k+mini_size] for k in xrange(0,n,mini_size)] #xrange 返回的是生成器
            for mini_batch in mini_batches:
                self.update_minibatch(mini_batch,eta)
            if test_data:
                print "epoches {0} acc {1}/{2} ".format(epoch,self.cal_acc(test_data),n_test)
            else:
                print "no test data"

    # mini_batch 代表mini_size 个元组(x,y) 其中x是输入的维度 ，输出是class的种类
    def update_minibatch(self,mini_batch,eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases] #随机梯度下降算法，用于存储我们指定的minibatch中的数据的bias的总和
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            new_nabla_b ,new_nabla_w = self.back(x,y)

            nabla_w = [nw+w for nw,w in zip(new_nabla_w,nabla_w)]
            nabla_b = [nb+b for nb,b in zip(new_nabla_b,nabla_b)]

        self.weights = [w-((eta/len(mini_batch))*nabla_w) for w,nabla_w in zip(self.weights,nabla_w)]
        self.biases = [b-((eta/len(mini_batch))*nabla_b) for b,nabla_b in zip(self.biases,nabla_b)]

    def back(self,x,y):

        nabla_b = [np.zeros(b.shape) for b in self.biases] #随机梯度下降算法，用于存储我们指定的minibatch中的数据的bias的总和
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation =x #第一层的激活值一般就是输入值，或者通过数据增强后的输入值
        activations = [x]
        zs = [] #存储中间结果，待激活值
        for w,b in zip(self.weights,self.biases):
           # print w.shape,b.shape
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = cost(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        #delta 目前可能是10*1 的向量，而activation[-2]可能是784*1 无法直接点乘 需要转置后者 最终是10个784维向量 内部相加输出 10*1
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.layers):
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime(zs[-l])
            nabla_b[-l] =delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)



    def cal_acc(self,test_data):
        test_set = [(self.feedforward(x),y) for x,y in test_data]
        return np.sum(int(np.argmax(x))==y for (x,y) in test_set)

    def feedforward(self,x):
        for w,b in zip(self.weights,self.biases):
            x = np.dot(w,x)+b
            x = sigmoid(x)  #一定不能忘记激活
        return x
def cost(x,y):
    #x is refer to the output of the network we deside!
    return x-y
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
def sigmoid(z):
    return 1/(1+np.exp(-z))

if __name__ == '__main__':

    train_data,validataion_data,test_data = mnist_loader.load_data_wrapper()

    #print train_data
    print type(train_data)

    net = cnn1([784,30,10])

    net.SGD(train_data,6,10,3.0,test_data=test_data)