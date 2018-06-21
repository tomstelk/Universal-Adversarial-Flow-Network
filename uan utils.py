import stadv
import  tensorflow as tf


def targetedAdvRate(target, perb_preds, orig_preds):
    
    fooled = perb_preds != orig_preds
    correctPerb = perb_preds == target
    nonTargetImages= orig_preds!=target
    
    advSuccess = fooled and correctPerb and nonTargetImages
    
    return  advSuccess.mean()

def calcPerbRate(X_test, Y_test, perbFlow, targeted, target, model, modelFile):
    
    
    saver = tf.train.Saver()
    
    images = tf.constant(X_test)
    perbFlow_tf = tf.constant(perbFlow)
    perb_images = stadv.layers.flow_st(images, perbFlow_tf)
    perb_preds = model.make_pred(perb_images)
    
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,modelFile)
        perb_images_np = sess.run(perb_preds, feed_dict = {images:X_test})
        
    
