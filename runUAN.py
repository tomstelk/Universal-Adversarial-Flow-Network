import tensorflow as tf
import stadv
import lossesLocal
import numpy as np
import uan_utils 
import datetime
from cleverhans.utils_mnist import data_mnist
from attackFlowUAN import attackFlowUAN
from tutorial_models import make_basic_cnn



def runUAN(paramDictFile, resDictFile, X_Train, Y_Train, X_Test, Y_Test, targetModelFunction, targetModelFile, oneHot=True):
    
    #Load
    paramDict = uan_utils.load_obj(paramDictFile)
    #Reset
    tf.reset_default_graph()
    
    #Target model
    targetModel = targetModelFunction()
    saver = tf.train.Saver()
    
    #placeholders
    images = tf.placeholder(tf.float32,(None, paramDict["imSize"][0], paramDict["imSize"][1], paramDict["nc"]))
    targets = tf.placeholder(tf.int64, [None])
    
    resDict = {"startTime": None,
           "endTime": None,
           "paramDict":None,
           "attackFlow": None,
           "testAdvRate": None,
           "trainAdvRate":None ,
           "trainPreds": None,
           "testPreds": None}
    
    #Convert Y labels from onehot 2 indices
    if oneHot:
        Y_Train = uan_utils.convertOneHot2Labels(Y_Train)
        Y_Test = uan_utils.convertOneHot2Labels(Y_Test)
    
    #Run UAN training
    resDict["startTime"]=datetime.datetime.now().time()
    uan = attackFlowUAN(paramDict)
    outAttackFlow, pred_orig, pred_perb, perturbed_images_np,advRateTrain =  uan.runTrain(X_Train, 
                        Y_Train, images, targets, saver, targetModel, targetModelFile)
    
    resDict["endTime"]=datetime.datetime.now().time()
    
    #Test attack flow on test data
    
    #Reset
    tf.reset_default_graph()
    #Target model
    targetModel = targetModelFunction()
    pred_perb_test = uan_utils.calcPerbPreds(X_Test, outAttackFlow, targetModel, targetModelFile)
    
    #Calc testAdvRate 
    orig_preds = uan_utils.calcPreds(X_Test, targetModel, targetModelFile)
    advRateTest = uan_utils.advRate(paramDict["targeted"], paramDict["advTarget"], pred_perb_test, orig_preds, Y_Test)
    
    resDict["paramDict"]=paramDict
    resDict["attackFlow"]=outAttackFlow
    resDict["trainAdvRate"]=advRateTrain
    resDict["trainPreds"]=pred_perb
    resDict["testPreds"]=pred_perb_test
    resDict["trainAdvRate"]=advRateTest
    #Save results
    uan_utils.save_obj(resDict, resDictFile)
    
    

#Define paramDict
paramDict = {"UANname": "attack",
"advTarget": 3,
"batchSize": 50,
"imFormat": "NHWC",
"imSize": (28,28),
"lFlowMax": 20,
"lrate": 0.01 ,
"n_epochs": 5,
"nc": 1,
"seed": 0 ,
"targeted":False ,
"tau": 0.05,
"trainSetSize": 100,
"trainsSetStart": 0,
"zSize": 100}


#Filenames
dict_dir = "C:/Users/user/Documents/NN Work/UAN/UAT test/"
paramDictFile =  dict_dir + "param_dict_" + uan_utils.getNowString(True) + ".pkl"
resDictFile = dict_dir + "res_dict_" + uan_utils.getNowString(True) + ".pkl"

#Save paramDict
save_obj(paramDict, paramDictFile)


# Get MNIST test data
MNIST_X_train, MNIST_Y_train, MNIST_X_test, MNIST_Y_test = data_mnist(train_start=0,
                                              train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)



#Target model file
modelfile  = "./savedmodels/basic_cnn_MNIST/basic_cnn.ckpt"


runUAN(paramDictFile, resDictFile, MNIST_X_train[0:100], MNIST_Y_train[0:100], MNIST_X_test, MNIST_Y_test, 
                 make_basic_cnn,modelfile, True)



