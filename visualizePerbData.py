import matplotlib.pyplot as plt
import stadv
import tensorflow as tf
def visualizePerb(attackFlow, xData):
    x_flow = attackFlow[0,0,:,:]
    y_flow = attackFlow[0,1,:,:]

    plt.rcParams['figure.figsize'] = [10, 10]
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    
    perbImages = stadv.layers.flow_st(xData, attackFlow)
    
    with tf.Session() as sess:
        perbImages_np = sess.run(perbImages)
    
    for i in range(xData.shape[0]):
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        
        x= xData[i,:,:,0]
        p= perbImages_np[i,:,:,0]
        ax1.imshow(x)
        ax1.axis('off')
        ax2.imshow(p)
        ax2.axis('off')
        ax3.imshow(x_flow)
        ax3.axis('off')
        ax4.imshow(y_flow)
        ax4.axis('off')
        plt.show()


    return perbImages_np