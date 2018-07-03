import pickle 
import numpy as np
import uan_utils
from sklearn import preprocessing
import matplotlib.pyplot as plt
from cleverhans.utils_mnist import data_mnist
def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def normDataSet(dataset, valMin, valMax):
    
    return (dataset-valMin)/(valMax-valMin)

def reshapeCIFAR(img_flat):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    
    return img


def reshapeMultiCIFAR(imgs_flat, numImages):
    
    out = np.zeros((numImages, 32, 32, 3))
    
    for i in range(0, numImages):
        
        out[i]=reshapeCIFAR(imgs_flat[i])
        
    return out 


def rawCIFAR10(dataDir, test = False):
    
    
    numImagesPreBatch = 10000
    if test:
        #Get test data
        cifarTestFile = dataDir + "test_batch.bin"
        cifarTestDict = unpickle(cifarTestFile)
                
        x = reshapeMultiCIFAR(cifarTestDict[b'data'], numImagesPreBatch)
        y = cifarTestDict[b'labels']
        
        
    else:
        
        #Get train data
        cifarTrainFile = dataDir + "data_batch_1.bin"
        cifarTrainDict = unpickle(cifarTrainFile)
        train_flat = cifarTrainDict[b'data']
        train  = reshapeMultiCIFAR(train_flat, numImagesPreBatch)
        Y_train =cifarTrainDict[b'labels']
        
        for i in range(2, 6):
            cifarTrainFile = dataDir + "data_batch_" + str(i) + ".bin"
            cifarTrainDict = unpickle(cifarTrainFile)

            x = np.concatenate((train,reshapeMultiCIFAR(cifarTrainDict[b'data'], numImagesPreBatch)))
            y =np.concatenate((Y_train, cifarTrainDict[b'labels']))
        
    return x, y

def load_data(dataset, dataDir, oneHot = False):
    
    if dataset=="CIFAR-10":
        
        # CIFAR-specific dimensions
        img_rows = 32
        img_cols = 32
        channels = 3
        num_classes = 10 
        
        train, Y_train = rawCIFAR10(dataDir, test = False)
        test, Y_test = rawCIFAR10(dataDir, test = True)
        
        #Norm data
        X_train = normDataSet(train,0,255)
        X_test = normDataSet(test,0,255)
        
        
        
        
        
        
    elif dataset=="MNIST":
        
        # CIFAR-specific dimensions
        img_rows = 28
        img_cols = 28
        channels = 1
        num_classes = 10 
        
        X_train, Y_train, X_test, Y_test = data_mnist()
        
        #Convert from onehot 2 index
        Y_train = uan_utils.convertOneHot2Labels(Y_train)
        Y_test = uan_utils.convertOneHot2Labels(Y_test)

    imsize = (img_rows, img_cols)
        
    return X_train, Y_train, X_test, Y_test, imsize, channels, num_classes



'''
# Get data
dataSet = "CIFAR-10"
dataSetDir = "../CIFAR-10/cifar-10-batches-py/"
modelfile  = "../savedmodels/basic_cnn_CIFAR10/basic_cnn_CIFAR10.ckpt"


X_train, Y_train, X_test, Y_test, imSize, nc, num_classes = load_data(dataSet, dataSetDir)
trainIdx=np.random.randint(0,50000)
testIdx=np.random.randint(0,10000)
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(X_train[trainIdx,:,:,:])
ax2.imshow(X_test[testIdx,:,:,:])


for i in range(0,10):
    f, (ax1) = plt.subplots(1, 1)
    x= X_train[i,:,:,:]
    ax1.imshow(x)
    
'''