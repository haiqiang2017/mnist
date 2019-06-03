import mnist_loader
import cnn

train_data,validataion_data,test_data = mnist_loader.load_data_wrapper()

#print train_data
print type(train_data)

net = cnn.cnn([784,30,10])

net.SGD(train_data,2,10,3.0,test_data=test_data)