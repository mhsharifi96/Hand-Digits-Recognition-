
# Handwritten Digit Recognition using neural network


# Requirements
* Python 3.5 +
* Numpy 
* cv2
* matplotlib
* tensorflow

# Usage
In this code use two dataset:

**1.** MNIST for recognize english digits

**2.** HODA for recognize persian digits

**3.** Install Requirements

### In NN dir
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

**1.** If you recognize english digits run ```python mnist_main.py```

**2.** If you recognize persian digits run ```python hoda_main.py```

### In tensorflow dir
Handwritten Digits with TensorFlow

**1.** If you recognize english digits run ```python ts_main.py```


# Result

## persian digits
```
$ python3 hoda_main.py 

Reading train dataset (Train 60000.cdb)...
Reading test dataset (Test 20000.cdb)...
Epoch 0 : 17882 / 20000 accuracy:89.41 
Epoch 1 : 18402 / 20000 accuracy:92.01 
Epoch 2 : 18584 / 20000 accuracy:92.92 
...

```
## english digits
```
$ python mnist_main.py

Epoch 0 : 9043 / 10000 accuracy:90.42
Epoch 1 : 9191 / 10000 accuracy:91.91 
Epoch 2 : 9285 / 10000 accuracy:92.85 
Epoch 3 : 9295 / 10000 accuracy:92.95 
Epoch 4 : 9383 / 10000 accuracy:93.83 
Epoch 5 : 9391 / 10000 accuracy:93.91
...

```

# reference
[Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)

[Hoda](http://farsiocr.ir/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%AF%D8%A7%D8%AF%D9%87/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%A7%D8%B1%D9%82%D8%A7%D9%85-%D8%AF%D8%B3%D8%AA%D9%86%D9%88%DB%8C%D8%B3-%D9%87%D8%AF%DB%8C/)

[tensorflow](https://www.tensorflow.org/)

[Handwritten Digits with TensorFlow](https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow)
