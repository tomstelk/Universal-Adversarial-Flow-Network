

import tensorflow as tf
from tutorial_models import make_basic_cnn
from loadData import load_data
import uan_utils
import matplotlib.pyplot as plt
import stadv
import numpy as np
import attackFlowUAN
import sys
import lossesLocal

'''
def Lflow(fl):
    fl_pad = np.pad(fl, [[0, 0], [0, 0], [1, 1], [1, 1]], 'edge')
    
    
    def _l2(t1, t2, axis):
        """Shortcut for getting the squared L2 norm of the difference
        between two tensors when slicing on the second axis.
        """

        return np.linalg.norm(t1[:, axis] - t2[:, axis], axis = (1,2))**2
    
    
    shifted_flows = [
            fl_pad[:, :, 2:, 2:],  # bottom right
            fl_pad[:, :, 2:, 1:-1],  # bottom mid
            fl_pad[:, :, 2:, :-2],  # bottom left
            fl_pad[:, :, 1:-1, :-2],  # mid left
            fl_pad[:, :, :-2, :-2],   # top left
            fl_pad[:, :, :-2, 1:-1],   # top mid
            fl_pad[:, :, :-2, 2:],  # top right
            fl_pad[:, :, 1:-1, 2:],  # mid right
            
        ]
    
    lf = sum([np.sqrt(_l2(fl,sf, 0)+_l2(fl,sf, 1)) for sf in shifted_flows])
    
    return lf
    



np.random.seed(0)
flow_np = np.random.rand(1,2,3,3)


test = Lflow(flow_np)

scale = 5.
testLina = Lflow(flow_np*scale)
testLinb = scale *test


x_flow_np=flow_np[0,0,:,:]
y_flow_np=flow_np[0,1,:,:]






flow=tf.constant(flow_np, dtype = tf.float32)

flowLoss_stadv = lossesLocal.flow_loss(flow,)

with tf.Session() as tf:
    flowLoss_stadv_np = flowLoss_stadv.eval()
    

x_flow_np=flow_np[0,0,:,:]
y_flow_np=flow_np[0,1,:,:]

x_padded_flow_np=padded_flow_np[0,0,:,:]
y_padded_flow_np=padded_flow_np[0,1,:,:]

x_shifted_flows_np=shifted_flows_np[0][0,0,:,:]
y_shifted_flows_np=shifted_flows_np[0][0,1,:,:]








#Test CIFAR model
dataSetDir = "../CIFAR-10/cifar-10-batches-py/"
targetModelFile  = "../savedmodels/basic_cnn_CIFAR10/basic_cnn_CIFAR10.ckpt"    
dictDir = "./res/CIFAR-10/"

#Reset
tf.reset_default_graph()

#Get relevant data
train, label, test, test_label, imsize, channels, num_classes = load_data("CIFAR-10", dataSetDir)


targetModel = make_basic_cnn(nb_classes = 10,  input_shape = (None, 32,32,3))

#cifarPredsTrain = uan_utils.calcPreds(train, targetModel, targetModelFile)
cifarPredsTest = uan_utils.calcPreds(test, targetModel, targetModelFile)


#UAN attack flow
num_classes=10
imSize = (32,32)
nc = 3
dataSet= ""

paramDict = {"UANname": "attack",
"advTarget": 3,
"batchSize": 100,
"imFormat": "NHWC",
"imSize": imSize,
"lFlowMax": 10,
"lrate": 0.01 ,
"n_epochs": 200,
"nc": nc,
"seed": 0 ,
"targeted":False ,
"tau": 0.001,
"trainSetSize": 10000,
"trainsSetStart": 0,
"zSize": 100,
"dataset": "ImageNet", 
"num_classes": num_classes,
"dataSet": dataSet,
"directed": True,
"angle": 0, 
"width": 1}


tf.reset_default_graph()
uan = attackFlowUAN.attackFlowUAN(paramDict)
attackFlowScaled, uan_var_list= uan.genAttackFlow()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    attackFlowScaled_np = sess.run([attackFlowScaled])
x_flow = attackFlowScaled_np[0][0,0,:,:]
y_flow = attackFlowScaled_np[0][0,1,:,:]

f, (ax1, ax2) = plt.subplots(1, 2)
        
ax1.imshow(x_flow)
ax1.axis('off')
ax2.imshow(y_flow)
ax2.axis('off')

'''


#Get results, visualise and compare preds

def visualizePerb(attackFlow, xData, save2File = False, outImDir = None, origPreds=None, perbPreds=None):
    x_flow = attackFlow[0,0,:,:]
    y_flow = attackFlow[0,1,:,:]

    plt.rcParams['figure.figsize'] = [10, 10]
    
    #Visualize x_flow and y_flow and save to file
    f, (ax1) = plt.subplots(1, 1)
    ax1.imshow(x_flow)
    ax1.axis('off')
    if save2File:
        plt.savefig(outImDir + "attackFlow_x.png")
    
    ax1.imshow(y_flow)
    ax1.axis('off')
    
    if save2File:
        plt.savefig(outImDir + "attackFlow_y.png")
    
    
    
    #Apply perturbation flow to x set
    perbImages = stadv.layers.flow_st(xData, attackFlow)
    
    
    
    with tf.Session() as sess:
        perbImages_np = sess.run(perbImages)
    
    
    for i in range(xData.shape[0]):
        perbImName="origPred_"  + str(origPreds[i]) + "_perbPred_"  + str(perbPreds[i]) + ".png"
        origImName="origPred_"  + str(origPreds[i]) + ".png"
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        
        x= xData[i,:,:,0]
        ax1.imshow(x)
        ax1.axis('off')
        
        if save2File:
            plt.savefig(outImDir + origImName)
        
        p= perbImages_np[i,:,:,0]
        ax2.imshow(p)
        ax2.axis('off')

        if save2File:
            plt.savefig(outImDir + perbImName)

    return perbImages_np




dataSet = "MNIST"

if dataSet == "MNIST":
    dataSetDir = ""
    targetModelFile  = "../savedmodels/basic_cnn_MNIST/basic_cnn.ckpt"
    dictDir = "./res/MNIST/squareLoss/targeted/lMax100/"
elif dataSet == "CIFAR-10":
    dataSetDir = "../CIFAR-10/cifar-10-batches-py/"
    targetModelFile  = "../savedmodels/basic_cnn_CIFAR10/basic_cnn_CIFAR10.ckpt"    
    dictDir = "./res/CIFAR-10/"





def analyseResDict(timeStamp, dataSet, dataSetDir, dictDir, save2File, 
                   outImDir):

    #Get relevant data
    train, label, test, test_label, imsize, channels, num_classes = load_data(dataSet, dataSetDir)

    resDictFile = dictDir + "res_dict_" + dataSet  + "_" + timeStamp + ".pkl"
    paramDictFile = dictDir + "param_dict_" + dataSet  +"_" +  timeStamp + ".pkl"
    resDict = uan_utils.load_obj(resDictFile)
    paramDict = uan_utils.load_obj(paramDictFile)
    
    attackFlow = resDict["attackFlow"]
    
    
    #choose a test subset
    subset= range(10)
    testSubSet =test[subset]


    #Reset
    tf.reset_default_graph()
        
    #Target model
    targetModel = make_basic_cnn(nb_classes = paramDict["num_classes"], 
                                      input_shape = (None, paramDict["imSize"][0], 
                                                     paramDict["imSize"][1], 
                                                     paramDict["nc"]))
    #Compare predictions
    origPredsSubset = uan_utils.calcPreds(testSubSet, targetModel, targetModelFile)
    perbPredsSubset = resDict["testPreds"][subset]

    #Visualize results
    visualizePerb(attackFlow, testSubSet, save2File, outImDir, origPredsSubset, perbPredsSubset)
    
    
    return origPredsSubset, perbPredsSubset, paramDict, resDict


'''
timeStamps = ["2018-06-30_11-32-14_t_0",
              "2018-06-30_12-28-12_t_1",
              "2018-06-30_13-23-10_t_2", 
              "2018-06-30_14-20-24_t_3", 
              "2018-06-30_15-14-33_t_4", 
              "2018-06-30_16-10-14_t_5", 
              "2018-06-30_17-02-47_t_6", 
              "2018-06-30_17-55-40_t_7", 
              "2018-06-30_18-48-24_t_8", 
              "2018-06-30_19-41-14_t_9"]

'''
timeStamps = ["2018-06-30_18-48-24_t_8"]
advTarget = []
testAdvRates = []
trainAdvRates = []

for ts in timeStamps:
    origPredsSubset, perbPredsSubset, paramDict, resDict = analyseResDict(ts, dataSet, dataSetDir, dictDir,
                                              False, dictDir)
    advTarget.append(paramDict["advTarget"])
    testAdvRates.append(resDict["testAdvRate"]*100)
    trainAdvRates.append(resDict["trainAdvRate"]*100)


''''
plt.bar(advTarget, trainAdvRates)

plt.xlabel("Target")
plt.ylabel("Adversarial Rate")
plt.xticks(advTarget)

plt.savefig(dictDir + "AdvRateVsTarget_LFlow100.png")

plt.subplot(2,1,1)  
plt.plot(lMaxs, trainAdvRates, 'o-')
plt.plot(lMaxs, testAdvRates, 'o-')

plt.ylabel('Adversarial Rate')
plt.xlabel('Lflow Max')
plt.legend(("Train", "Test"))

plt.savefig(dictDir + "AdvRateVsLflowMax.png")
'''