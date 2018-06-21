import stadv
import  tensorflow as tf




def targetedAdvRate(target, perb_preds, orig_preds, groundTruth):
        
    succCount = 0
    potentialCount =0
    for i in range(orig_preds.shape[0]):
        if orig_preds[i]!=target and orig_preds[i] == groundTruth[i]:
            potentialCount = potentialCount+1
            if orig_preds[i]!=perb_preds[i] and perb_preds[i]== target:
                succCount=succCount+1
            
    return  round(succCount/potentialCount,2)


def unTargetedAdvRate(perb_preds, orig_preds,groundTruth):
    
    succCount = 0
    potentialCount =0
    for i in range(orig_preds.shape[0]):
        if orig_preds[i] == groundTruth[i]:
            potentialCount = potentialCount+1
            if orig_preds[i]!=perb_preds[i]:
                succCount=succCount+1
            
    return  round(succCount/potentialCount,3)

def advRate(targeted, target, perb_preds, orig_preds, groundTruth):
    if targeted: 
        return targetedAdvRate(target, perb_preds, orig_preds, groundTruth)
    else:
        return unTargetedAdvRate(perb_preds, orig_preds, groundTruth)

def calcPerbPreds(X_testData, perbFlow, model, modelFile):
    
    saver = tf.train.Saver()
    images = tf.constant(X_testData)
    perbFlow_tf = tf.constant(perbFlow)
    perb_images = stadv.layers.flow_st(images, perbFlow_tf)
    perb_preds = model.make_pred(perb_images)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,modelFile)
        perb_preds_np = sess.run(perb_preds, feed_dict = {images:X_testData})
    
    return perb_preds_np



def calcPreds(X_testData, model, modelFile):
    
    saver = tf.train.Saver()
    images = tf.constant(X_testData)
    
    preds = model.make_pred(X_testData)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,modelFile)
        preds_np = sess.run(preds, feed_dict = {images:X_testData})
    
    return preds_np
    '''
def calcAdvRateFromPerb(X_test, Y_test, perbFlow, targeted, target, model, modelFile):
    
    
    saver = tf.train.Saver()
    
    images = tf.constant(X_test)
    perbFlow_tf = tf.constant(perbFlow)
    perb_images = stadv.layers.flow_st(images, perbFlow_tf)
    perb_preds = model.make_pred(perb_images)
    orig_preds = model.make_pred(images)
    
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,modelFile)
        perb_preds_np = sess.run(perb_preds, feed_dict = {images:X_test})
        orig_preds_np = sess.run(orig_preds, feed_dict = {images:X_test})
    
    return advRate(targeted, target, perb_preds_np,orig_preds_np, Y_test)

'''