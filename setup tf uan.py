


import numpy as np
from cleverhans.utils_mnist import data_mnist
from saveloadPickle import save_obj, load_obj
from uanCalcAttackFlow import calcAttackFlow
import matplotlib.pyplot as plt
import visualizePerbData
import uan_utils
import tensorflow as tf
import tutorial_models
import datetime
#Parameters




paramDict={"trainSetSize": 10000,
    "trainsSetStart": 0,
    "batchSize": 100,
    "n_epochs": 25,
    "advTarget": 1,
    "targeted": True,
    "modelFile": "./savedmodels/basic_cnn_MNIST/basic_cnn.ckpt",
    "lFlowMax": 25,
    "UANname": "attack",
    "imFormat": "NHWC",
    "seed": 0,
    "zSize": 100,
    "lrate": 0.01,
    "tau": 0.05}



resDir = "C:/Users/user/Documents/NN Work/UAN/UAN_run1/"
resultDictFileName = "resultDict_"

resDict = {"startTime": None,
           "endTime": None,
           "paramDict":None,
           "attackFlow": None,
           "testAdvRate": None,
           "trainAdvRate":None ,
           "trainPreds": None,
           "testPreds": None}

#Load MNIST
train_start=0
train_end=60000
test_start=0
test_end=10000

# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

for i in range(10):
    paramDict["advTarget"] = i
    
    #Reset
    tf.reset_default_graph()

    resDict["startTime"]=datetime.datetime.now().time()
    outAttackFlow, TrainPredOrig, TrainPredPerb, perturbed_images_np = calcAttackFlow(paramDict,X_train,Y_train)
    
    
    Y_Index = [np.where(r==1)[0][0] for r in 
               Y_train[paramDict["trainsSetStart"]:
                   paramDict["trainsSetStart"]+paramDict["trainSetSize"]]]
    
    trainAdvRate = uan_utils.advRate(paramDict["targeted"], paramDict["advTarget"], 
                                 TrainPredPerb, TrainPredOrig,    Y_Index)

    #Reset
    tf.reset_default_graph()

    model = tutorial_models.make_basic_cnn()
    testPerbPreds = uan_utils.calcPerbPreds(X_test, outAttackFlow, model, paramDict["modelFile"])
    testOrigPreds = uan_utils.calcPreds(X_test, model, paramDict["modelFile"])

    
    Y_Index_test = [np.where(r==1)[0][0] for r in Y_test]
    
    testAdvRate = uan_utils.advRate(paramDict["targeted"], paramDict["advTarget"], 
                                 testPerbPreds, testOrigPreds, Y_Index_test)
    
    
    resDict["endTime"]=datetime.datetime.now().time()
    resDict["paramDict"]=paramDict
    resDict["attackFlow"]=outAttackFlow
    resDict["testAdvRate"]=testAdvRate
    resDict["trainAdvRate"]=trainAdvRate
    resDict["trainPreds"]=TrainPredPerb
    resDict["testPreds"]=testPerbPreds
    
                                
    save_obj( resDict,resDir + resultDictFileName + "target_" + str(i) + '.pkl')
    
