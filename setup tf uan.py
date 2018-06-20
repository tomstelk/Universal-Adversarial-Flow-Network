


import numpy as np
from cleverhans.utils_mnist import data_mnist
from saveloadPickle import save_obj, load_obj
from uanCalcAttackFlow import calcAttackFlow

#Parameters



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

testRes = calcAttackFlow(paramDict,X_train,Y_train)


