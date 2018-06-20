import tensorflow as tf
import stadv
import lossesLocal
import numpy as np

class attackFlowUAN():
    
    def __init__(self,lFlowMax, targetModel, zSize =100, imSize = (28,28), UANname = 'attack', imFormat = 'NHWC', nc=1, targeted=False, advTarget =0, seed=0):
        
        self.targetModel = targetModel
        self.zSize =zSize
        self.imSize =imSize
        self.UANname = UANname
        self.imFormat = imFormat
        self.nc = nc
        self.lFlowMax= lFlowMax
        self.advTarget= advTarget
        self.targeted = targeted
        self.seed = seed
        self.trained = False
        
        
    def genAttackFlow(self):
        
        pre_uan_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        tf.set_random_seed(self.seed)
        z = tf.random_normal([1,1,1,self.zSize])
        #Layer1
        dc1=tf.layers.conv2d_transpose(z, 1, 256,strides=3,name = self.UANname + '_dc1')
        bn1=tf.layers.batch_normalization(dc1, name=self.UANname + '_bn1')
        r1 = tf.nn.relu(bn1)
        
        #Layer2
        '''
        dc2=tf.layers.conv2d_transpose(r1, 1, 128,strides=5,name = self.UANname + '_dc2')
        bn2=tf.layers.batch_normalization(dc2, name=self.UANname + '_bn2')
        r2 = tf.nn.relu(bn2)
        '''
        
        flat1 = tf.layers.flatten(r1, name=self.UANname + '_flat1')
        fc1 = tf.contrib.layers.fully_connected(flat1, 2*self.imSize[0]*self.imSize[1],  activation_fn=None)
        attackFlow = tf.reshape(fc1,[1,2,self.imSize[0],self.imSize[1]])
            
        scale = self.lFlowMax/stadv.losses.flow_loss(attackFlow)
        attackFlowScaled=attackFlow*scale
        
        all_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        uan_var_list = [v for v in all_var_list if v not in pre_uan_var_list]
        
        return attackFlowScaled, uan_var_list
    
    def createTrainOp(self, tau, lrate, images, targets):
    

        tau = tf.constant(tau, dtype=tf.float32, name='tau')        
        attackFlow, uan_var_list = self.genAttackFlow()
        
        perturbed_images = stadv.layers.flow_st(images, attackFlow, 'NHWC')
        logits = self.targetModel.get_logits(perturbed_images)
        
        if self.targeted:        
            L_adv = lossesLocal.adv_loss(logits, targets)
        else:
            L_adv = lossesLocal.adv_loss_untargeted(logits, targets)
                
        L_flow = stadv.losses.flow_loss(attackFlow, padding_mode='CONSTANT')
        L_final = L_adv + tau * L_flow

        optim=tf.train.GradientDescentOptimizer(learning_rate = lrate)
        train_op = optim.minimize(L_final, var_list = uan_var_list)
        
        return attackFlow, train_op
    
    def runTrain(self,images_np, labels_np, n_epochs, batchSize,images, 
                 targets,  targetModelFile , train_op,saver,attackFlow):
        
        trainSetSize = images_np.shape[0]
        numBatches = int(np.ceil(trainSetSize/batchSize))
        
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,targetModelFile)
        
    
            for i in range(n_epochs):
        
                for b in range(numBatches):
                    test_image=images_np[b*batchSize:(b+1)*batchSize]
                    test_label = labels_np[b*batchSize:(b+1)*batchSize]

                    sess.run(train_op, feed_dict = {images:test_image,targets: test_label})
                    
                perturbed_images = stadv.layers.flow_st(images_np,attackFlow)
                pred1=sess.run(self.targetModel .make_pred(images), feed_dict = {images:images_np})
                pred2=sess.run(self.targetModel .make_pred(perturbed_images), feed_dict = {images:images_np})
                print ('Epoch: ' + str (i) + ', adv rate: ' + str(100*np.count_nonzero(pred2-pred1)/len(pred1)) + '%')
                
                
            outAttackFlow = sess.run(attackFlow)
            
        return outAttackFlow

    
        