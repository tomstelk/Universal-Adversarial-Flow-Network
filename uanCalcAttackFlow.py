
import tensorflow as tf
import numpy as np

import attackFlowUAN 
# dependencies specific to this demo notebook
import matplotlib.pyplot as plt
from tutorial_models import make_basic_cnn
from cleverhans.utils_mnist import data_mnist
from saveloadPickle import save_obj, load_obj



def calcAttackFlow(paramDict, X, Y):

    
    #Set Params
    trainSetSize=paramDict["trainSetSize"]
    trainsSetStart=paramDict["trainsSetStart"]
    batchSize = paramDict["batchSize"]
    n_epochs=paramDict["n_epochs"]
    advTarget = paramDict["advTarget"]
    targeted = paramDict["targeted"]
    cnnModelFile = paramDict["modelFile"]
    lFlowMax = paramDict["lFlowMax"]
    UANname=paramDict["UANname"]
    imFormat = paramDict["imFormat"]
    seed = paramDict["seed"]
    zSize = paramDict["zSize"]
    lrate = paramDict["lrate"]
    tau = paramDict["tau"]


    #Select UAN attack training data
    X_TrainSubset = X[trainsSetStart:trainsSetStart+trainSetSize]
    Y_TrainSubset = [np.where(r==1)[0][0] for r in Y[trainsSetStart:trainsSetStart+trainSetSize]]

    #Data params
    imSize = (X.shape[1],X.shape[2])
    nc = X.shape[3]



    #Set Target 
    target_np = advTarget*np.ones((batchSize))


    #Reset
    tf.reset_default_graph()

    #Setup CNN model
    model = make_basic_cnn()
    saver = tf.train.Saver()
    

    #placeholders
    images = tf.placeholder(tf.float32,(None, imSize[0], imSize[1], nc))
    targets = tf.placeholder(tf.int64, [None])

    #UAN definition
    uan = attackFlowUAN.attackFlowUAN(lFlowMax, model, zSize, imSize, UANname, imFormat, nc, targeted, advTarget, seed)

    #Train op
    attackFlow, train_op = uan.createTrainOp(tau,lrate, images, targets)


    #Run training
    attack_np=uan.runTrain(X_TrainSubset, Y_TrainSubset, n_epochs, batchSize,images, targets,
             cnnModelFile , train_op,saver, attackFlow)


    return attack_np