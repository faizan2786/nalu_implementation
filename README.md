# nalu_implementation
The research engineers at DeepMind including well known AI researcher, Andrew Trask have published an impressive paper on a 
neural network model that can learn simple to complex numerical functions with great extrapolation (generalisation) ability.

![alt text](https://3c1703fe8d.site.internapcdn.net/newman/gfx/news/2018/1-deepmindrese.jpg "NALU diagram")

This repository is created to show the Neural Arithmetic Logic Unit (**NALU**) implementation in python using **Tensorflow**. 
The code in this repo complements my article on Medium on [NALUs](https://medium.com/@faizanmukardam/simple-guide-to-neural-arithmetic-logic-units-nalu-explanation-intuition-and-code-64bc22605712/). 
If you are new to NALUs, I strongly recommend my post for simple and intuitive explanation.

I have added three python files here:
* **NAC_simple_implementation.py** -> It implements the simple Neural Accumulator in python that is able to learn the addition and subtraction functions.
* **NAC_Complex_implementation.py** -> It implements the complex Neural Accumulator architecture in python that is able to learn the complex arithmetic functions such as multiplication, division and power function.
* **nalu_implementation.py** -> It implements the complete NALU using the same NAC implementation given in previous two files. 

**Note**: I have also added the code for training and evaluation on test data in each file for completeness

I use the *random uniform intiializer* to initialize unconstrained parametrs W_hat and M_hat within the range [-2,2] . However one may use any recommended weight initializers as he want.
```python

W_hat = tf.get_variable(name = "W_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),shape=[in_features, out_units],  trainable=True)

M_hat = tf.get_variable(name = "M_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2), shape=[in_features, out_units], trainable=True)
```
For training and test data I generate the series of integers for both input x1 and x2 using numpy's arange API.

```python
# Generate a series of input number X1 and X2 for training
x1 = np.arange(0,10000,5, dtype=np.float32)
x2 = np.arange(5,10005,5, dtype=np.float32)

# Prepare the input vector for training
x_train = np.column_stack((x1,x2))

# Generate a series of input number X1 and X2 for testing
x1 = np.arange(1000,2000,8, dtype=np.float32)
x2 = np.arange(1000,1500,4, dtype= np.float32)

# Prepare the input vector for testing
x_test = np.column_stack((x1,x2)

```
For the **simple NAC** I evaluate the addition operation on the generated training data `y = x1 + x2`. Whereas, for the **complex NAC**
I use simple multiplication function `y = x1 * x2` to evaluate the netwrok. 
For NALU, I use the complex numeric function `y = (x1/4) + (x2/2) + x3**2` to evaluate the network.
