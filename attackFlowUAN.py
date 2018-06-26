import tensorflow as tf
import stadv
import lossesLocal
import numpy as np
import uan_utils 
class attackFlowUAN():
    
    def __init__(self, paramDict):
        
        
        
        self.batchSize = paramDict["batchSize"]
        self.n_epochs=paramDict["n_epochs"]
        self.advTarget = paramDict["advTarget"]
        self.targeted  = paramDict["targeted"]
        self.lFlowMax = paramDict["lFlowMax"]
        self.UANname = paramDict["UANname"]
        self.imFormat = paramDict["imFormat"]
        self.seed = paramDict["seed"]
        self.zSize = paramDict["zSize"]
        self.lrate = paramDict["lrate"]
        self.tau = paramDict["tau"]
        self.imSize =paramDict["imSize"]
        self.nc = paramDict["nc"]
        
        
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
    
        
    def runTrain(self,images_np, labels_np,  images, targets,  saver, targetModel, targetModelFile):

        #Size data
        trainSetSize = images_np.shape[0]
        numBatches = int(np.ceil(trainSetSize/self.batchSize))
        
        #Optimizatio param
        tau_tf = tf.constant(self.tau, dtype=tf.float32, name='tau')        
        
        #Define attack flow field via UAN, and list UAN variables to opt over
        attackFlow, uan_var_list = self.genAttackFlow()
        
        #Apply flow field and calculate logits
        perturbed_images = stadv.layers.flow_st(images, attackFlow, self.imFormat)
        logits = targetModel.get_logits(perturbed_images)
        
        #If targetd use the adv target label, else use correct labels to opt away from
        if self.targeted:        
            L_adv = stadv.losses.adv_loss(logits, targets)
            trainTargets = self.advTarget*np.ones(images_np.shape[0])
        else:
            L_adv = lossesLocal.adv_loss_untargeted(logits, targets)
            trainTargets = labels_np            
            
            
        #Define loss function
        L_flow = stadv.losses.flow_loss(attackFlow, padding_mode='CONSTANT')
        L_final = L_adv + tau_tf * L_flow

        #Define optimization
        optim=tf.train.GradientDescentOptimizer(learning_rate = self.lrate)
        train_op = optim.minimize(L_final, var_list = uan_var_list)
        
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,targetModelFile)
        
    
            for i in range(self.n_epochs):
        
                for b in range(numBatches):
                    test_image=images_np[b*self.batchSize:(b+1)*self.batchSize]
                    test_label = trainTargets[b*self.batchSize:(b+1)*self.batchSize]

                    sess.run(train_op, feed_dict = {images:test_image,targets: test_label})
                    
                
                pred_orig=sess.run(targetModel .make_pred(images), feed_dict = {images:images_np})
                pred_perb=sess.run(targetModel .make_pred(perturbed_images), feed_dict = {images:images_np})
                
                print ('Epoch: ' + str (i) + ', adv rate: ' + 
                       str(100*uan_utils.advRate(self.targeted, self.advTarget,pred_perb, pred_orig,labels_np )) + '%')
                
            #Output attack flow field, old predictions, new predictions, and perturbed images
            outAttackFlow = sess.run(attackFlow)
            pred_orig, pred_perb, perturbed_images_np =sess.run(
                    [targetModel .make_pred(images), 
                     targetModel .make_pred(perturbed_images), 
                     perturbed_images], feed_dict = {images:images_np})
            
            advRateTrain = uan_utils.advRate(self.targeted, self.advTarget,pred_perb, pred_orig,labels_np )
            
        return outAttackFlow, pred_orig, pred_perb, perturbed_images_np,advRateTrain

    
        