
from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import network
import numpy as np
from pprint import pprint
from random import sample
from HodaDatasetReader import read_hoda_dataset

from PIL import Image

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    j=int(j)
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():

    print('Reading train dataset (Train 60000.cdb)...')
    X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                    images_height=28,
                                    images_width=28,
                                    one_hot=False,
                                    reshape=True)

    training_inputs = [np.reshape(x, (784,1)) for x in X_train]
    training_results = [vectorized_result(y) for y in Y_train]
    training_data = zip(training_inputs, training_results)


    print('Reading test dataset (Test 20000.cdb)...')
    X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                images_height=28,
                                images_width=28,
                                one_hot=False,
                                reshape=True)
    
    test_inputs = [np.reshape(x, (784, 1)) for x in X_test]
    Y_test=[int(y) for y in Y_test]
    test_data = zip(test_inputs, Y_test)

    print('Reading remaining samples dataset (RemainingSamples.cdb)...')
    X_remaining, Y_remaining = read_hoda_dataset('./DigitDB/RemainingSamples.cdb',
                                             images_height=28,
                                             images_width=28,
                                             one_hot=False,
                                             reshape=True)
    validation_inputs = [np.reshape(x, (784, 1)) for x in X_remaining]
    validation_data = zip(validation_inputs, Y_remaining)
    
    return (training_data,validation_data,test_data)

training_data,validation_data,test_data =load_data_wrapper()
#  
# i665=1
# tra_1=[]
# for (x,y) in training_data:
#     if i665==1:
#         tra_1=[x,y]
#     i665=i665+1
# print(tra_1)
# # i=1
# # for (x,y) in test_data:
# #     if i==8753:
# #         print(x,y)
# #     i=i+1


net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# net.SGD(training_data, 30, 10, 3.0, valid_data=validation_data)
# validation_data=list(validation_data)
# plots_x = 1
# plots_y = 1
# fig = plt.figure(figsize=(32, 8))
# samples = sample(range(20000), k=plots_x*plots_y)
# samples.sort()
# counter = 0
# print(validation_data[0][0])
# # plt.title('Y_remaining[ ' + str(samples[counter]) + ' ], E: ' + str(list(Y_remaining[samples[counter]]).index(1)) + ', P: ' + str(np.squeeze(prediction)))
# plt.imshow(validation_data[0][0].reshape([18, 18]), cmap='gray')

# net.SGD(training_data, 30, 10, 3.0)
