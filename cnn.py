# _*_ coding:utf-8 _*_
import numpy as np
import random
import sys
import mnist_loader
class cnn(object):


    #sizes is a list of parameters
    def __init__(self,sizes):
        self.layer_numbers = len(sizes)
        #第二层以后的每个神经元都有一个矩阵


        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,train_data,epoches,mini_batch_size,eta,test_data=None):
        if test_data: n_test = len(test_data)

        n = len(train_data)
        for epoch in range(epoches):
           random.shuffle(train_data)
           mini_batches = [train_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
           # for i in range(0,n,mini_batch_size):
           #     mini_batch.append(train_data[i:i+mini_batch_size])
           print len(mini_batches)
           for mini_batch in mini_batches:
               self.update_mini_batch(mini_batch,eta)
           if test_data:
               print "Epoches {0}:{1}/{2}".format(epoch,self.evaluate(test_data),n_test)
           else:
               print "Epoches {0} is done".format(epoch)

    def update_mini_batch(self,mini_batch,eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)

            nabla_b = [nb+nwb for nb,nwb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+nww for nw,nww in zip(nabla_w,delta_nabla_w)]

        # self.weights = [w-(eta/len(mini_batch))*nabla_w for w,nabla_w in zip(self.weights,nabla_w)]
        # self.biases = [b-(eta/len(mini_batch)*nabla_b) for b,nabla_b in zip(self.biases,nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # 评测程序

    def evaluate(self,test_data):
        test_result = [(np.argmax(self.feedforward(x)),y) for x,y in test_data]
        return sum(int(x==y) for (x,y) in test_result)

    def backprop(self,x,y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x #初始化的avtivation是输入的向量
        activations = [x]
        zs = []
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = cost(y,activations[-1])*sigmoid_primer(zs[-1])


        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())



        for i in range(2,self.layer_numbers):
            delta = np.dot(self.weights[-i+1].transpose(),delta)*sigmoid_primer(zs[-i])
            print delta.shape
            print activations[-i-1].transpose().shape
            sys.exit(0)
            nabla_b[-i]=delta
            nabla_w[-i] = np.dot(delta,activations[-i-1].transpose())
        return (nabla_b,nabla_w)
    def backprop1(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            #x相应位置相乘求和
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass #计算最后一层的error
        delta = cost(y,activations[-1]) * sigmoid_primer(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #哈哈
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.layer_numbers):
            z = zs[-l]
            sp = sigmoid_primer(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #嘿嘿
        #返回一个实例的bios更新值和w的更新值
        return (nabla_b, nabla_w)

def sigmoid_primer(x):
    return sigmoid(x)*(1-sigmoid(x))
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def cost(y,a):
    return  a-y


if __name__ == '__main__':
    train_data, validataion_data, test_data = mnist_loader.load_data_wrapper()
    print len(test_data)
    # print train_data
    print type(train_data)

    net = cnn([784, 30, 10])

    net.SGD(train_data, 5, 10, 3.0, test_data=test_data)