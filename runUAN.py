import tensorflow as tf
import numpy as np
import uan_utils 
import datetime
from cleverhans.utils_mnist import data_mnist
from attackFlowUAN import attackFlowUAN
from tutorial_models import make_basic_cnn
from loadData import load_data


def runUAN(paramDictFile, resDictFile, X_Train, Y_Train, X_Test, Y_Test,
           targetModelFunc, targetModelFile, oneHot):
    
    #Load
    paramDict = uan_utils.load_obj(paramDictFile)
    #Reset
    tf.reset_default_graph()
    
    #Target model
    targetModel = targetModelFunc(nb_classes = paramDict["num_classes"], 
                                  input_shape = (None, paramDict["imSize"][0], 
                                                 paramDict["imSize"][1], 
                                                 paramDict["nc"]))
    saver = tf.train.Saver()
    
    #placeholders
    images = tf.placeholder(tf.float32,
                            (None, paramDict["imSize"][0], paramDict["imSize"][1], 
                             paramDict["nc"]))
    
    targets = tf.placeholder(tf.int64, [None])
    
    resDict = {"startTime": None,
           "endTime": None,
           "paramDict":None,
           "attackFlow": None,
           "testAdvRate": None,
           "trainAdvRate":None ,
           "trainPreds": None,
           "testPreds": None,
           "attackLFlow": None}
    
    #Convert Y labels from onehot 2 indices
    if oneHot:
        Y_Train = uan_utils.convertOneHot2Labels(Y_Train)
        Y_Test = uan_utils.convertOneHot2Labels(Y_Test)
    
    #Run UAN training
    resDict["startTime"]=datetime.datetime.now().time()
    uan = attackFlowUAN(paramDict)
    outAttackFlow, pred_orig, pred_perb, perturbed_images_np,advRateTrain, attackLFlow =  uan.runTrain(X_Train, 
                        Y_Train, images, targets, saver, targetModel, targetModelFile)
    
    resDict["endTime"]=datetime.datetime.now().time()
    
    #Test attack flow on test data
    
    #Reset
    tf.reset_default_graph()
    
    #Target model
    targetModel = targetModelFunc(nb_classes = paramDict["num_classes"], 
                                  input_shape = (None, paramDict["imSize"][0], 
                                                 paramDict["imSize"][1], 
                                                 paramDict["nc"]))
    
    pred_perb_test = uan_utils.calcPerbPreds(X_Test, outAttackFlow, targetModel, targetModelFile)
    
    #Calc testAdvRate 
    orig_preds = uan_utils.calcPreds(X_Test, targetModel, targetModelFile)
    advRateTest = uan_utils.advRate(paramDict["targeted"], paramDict["advTarget"], pred_perb_test, orig_preds, Y_Test)
    
    resDict["paramDict"]=paramDict
    resDict["attackFlow"]=outAttackFlow
    resDict["trainAdvRate"]=advRateTrain
    resDict["trainPreds"]=pred_perb
    resDict["testPreds"]=pred_perb_test
    resDict["testAdvRate"]=advRateTest
    resDict["attackLFlow"] = attackLFlow
    #Save results
    uan_utils.save_obj(resDict, resDictFile)
    



# Get data
dataSet = "MNIST"

if dataSet == "MNIST":
    dataSetDir = ""
    modelfile  = "../savedmodels/basic_cnn_MNIST/basic_cnn.ckpt"
elif dataSet == "CIFAR-10":
    dataSetDir = "../CIFAR-10/cifar-10-batches-py/"
    modelfile  = "../savedmodels/basic_cnn_CIFAR10/basic_cnn_CIFAR10.ckpt"    


X_train, Y_train, X_test, Y_test, imSize, nc, num_classes = load_data(dataSet, dataSetDir)



#Define paramDict
paramDict = {"UANname": "attack",
"advTarget": 0,
"batchSize": 128,
"imFormat": "NHWC",
"imSize": imSize,
"lFlowMax": 100,
"lrate": 0.01 ,
"n_epochs": 100,
"nc": nc,
"seed": 0 ,
"targeted":True ,
"tau": 0.001,
"trainSetSize": 10000 ,
"trainsSetStart": 0,
"zSize": 100,
"dataset": dataSet, 
"num_classes": num_classes,
"dataSet": dataSet,
"directed": False,
"angle": 0, 
"width": 1,
"squareLoss": True}




for t in range(10):

    #Filenames
    dict_dir = "./res/" + dataSet + "/squareLoss/targeted/lMax100/"
    paramDictFile =  dict_dir + "param_dict_" + dataSet + "_" + uan_utils.getNowString(True) + "_t_" + str(t) + ".pkl"
    resDictFile = dict_dir + "res_dict_" + dataSet  + "_" + uan_utils.getNowString(True) + "_t_" + str(t) + ".pkl"

    paramDict["advTarget"] = t

    #Save paramDict
    uan_utils.save_obj(paramDict, paramDictFile)



    t0 = paramDict["trainsSetStart"]
    tEnd = paramDict["trainsSetStart"]+paramDict["trainSetSize"]


    runUAN(paramDictFile, resDictFile, X_train[t0:tEnd], Y_train[t0:tEnd], X_test, Y_test, 
                 make_basic_cnn,modelfile, False)


