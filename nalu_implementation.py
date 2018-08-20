import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Define a Neural Accumulator (NAC) for addition/subtraction -> Useful to learn the addition/subtraction operation

def nac_simple_single_layer(x_in, out_units):
    '''
    Define a Neural Accumulator (NAC) for addition/subtraction -> Useful to learn the addition/subtraction operation

    Attributes:
        x_in -> Input vector
        out_units -> number of output neurons

    Return:
        Output tensor of mentioned shsape and associated weights
    '''

    in_features = x_in.shape[1]

    # define W_hat and M_hat

    W_hat = tf.get_variable(name = "W_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),shape=[in_features, out_units],  trainable=True)
    M_hat = tf.get_variable(name = "M_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2), shape=[in_features, out_units], trainable=True)

    # Get W

    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    y_out = tf.matmul(x_in,W)

    return y_out,W

# define a complex nac in log space -> for more complex arithmetic functions such as
# multiplication, division and power

def nac_complex_single_layer(x_in, out_units, epsilon = 0.000001):

    '''
    :param x_in: input feature vector
    :param out_units: number of output units of the cell
    :param epsilon: small value to avoid log(0) in the output result
    :return: associated weight matrix and output tensor
    '''

    in_shape = x_in.shape[1]

    W_hat = tf.get_variable(shape=[in_shape, out_units],
                            initializer= tf.initializers.random_uniform(minval=-2, maxval=2),
                            trainable=True, name="W_hat2")

    M_hat = tf.get_variable(shape=[in_shape, out_units],
                            initializer=tf.initializers.random_uniform(minval=-2, maxval=2),
                            trainable=True, name="M_hat2")

    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    # Express Input feature in log space to learn complex functions
    x_modified = tf.log(tf.abs(x_in) + epsilon)

    m = tf.exp( tf.matmul(x_modified, W) )

    return m, W

# Define a NALU having combination of NAC1 and NAC2

def nalu(x_in, out_units, epsilon=0.000001, get_weights=False):
    '''
    :param x_in: input feature vector
    :param out_units: number of output units of the cell
    :param epsilon: small value to avoid log(0) in the output result
    :param get_weights: True if want to get the weights of the model
                        in return
    :return: output tensor
    :return: Gate weight matrix
    :return: NAC1 (simple NAC) weight matrix
    :return: NAC2 (complex NAC) weight matrix
    '''

    in_shape = x_in.shape[1]

    # Get output of NAC1
    a, W_simple = nac_simple_single_layer(x_in, out_units)

    # Get output of NAC2
    m, W_complex = nac_complex_single_layer(x_in, out_units, epsilon= epsilon)

    # Gate signal layer
    G = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0),
                        shape=[in_shape, out_units], name="Gate_weights")

    g =  tf.nn.sigmoid( tf.matmul(x_in, G) )

    y_out = g * a + (1 - g) * m

    if(get_weights):
        return y_out, G, W_simple, W_complex
    else:
        return y_out


# Test the Network by learning the adition

# Generate a series of input number X1,X2 and X3 for training
x1 =  np.arange(1000,11000, step=5, dtype= np.float32)
x2 =  np.arange(500, 6500 , step=3, dtype= np.float32)
x3 = np.arange(0, 2000, step = 1, dtype= np.float32)


# Make any function of x1,x2 and x3 to try the network on
y_train = (x1/4) + (x2/2) + x3**2

x_train = np.column_stack( (x1,x2,x3) )

print(x_train.shape)
print(y_train.shape)

# Generate a series of input number X1,X2 and X3 for testing
x1 =  np.random.randint(0,1000, size= 200).astype(np.float32)
x2 = np.random.randint(1, 500, size=200).astype(np.float32)
x3 = np.random.randint(50, 150 , size=200).astype(np.float32)

x_test = np.column_stack((x1,x2,x3))
y_test = (x1/4) + (x2/2) + x3**2

print()
print(x_test.shape)
print(y_test.shape)


# Define the placeholder to feed the value at run time
X = tf.placeholder(dtype=tf.float32, shape =[None , 3])    # Number of samples x Number of features (number of inputs to be added)
Y = tf.placeholder(dtype=tf.float32, shape=[None,])

# define the network
# Here the network contains only one NAC cell (for testing)
y_pred  = nalu(X, out_units=1)
y_pred = tf.squeeze(y_pred)             # Remove extra dimensions if any

# Mean Square Error (MSE)
loss = tf.reduce_mean( (y_pred - Y) **2)
#loss= tf.losses.mean_squared_error(labels=y_train, predictions=y_pred)



# training parameters
alpha = 0.005 # learning rate
epochs = 10000
batch_size = 128

train_itr = np.int32(np.ceil( float(x_train.shape[0])/batch_size ))

optimize = tf.train.AdadeltaOptimizer(learning_rate=alpha).minimize(loss)

with tf.Session() as sess:

    #init = tf.global_variables_initializer()
    cost_history = []

    sess.run(tf.global_variables_initializer())

    # pre training evaluate
    print("Pre training MSE: ", sess.run (loss, feed_dict={X: x_test, Y:y_test}))
    print()
    for i in range(epochs):
        epoch_cost = 0
        for itr in range(train_itr):
            _, cost = sess.run([optimize, loss ], feed_dict={X:x_train, Y: y_train})
            epoch_cost += cost

        mse = epoch_cost/train_itr
        print("epoch: {}, MSE: {}".format( i, mse) )
        cost_history.append(mse)

    # plot the MSE over each iteration
    plt.plot(np.arange(epochs),np.log(cost_history))  # Plot MSE on log scale
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()

    print()
    #print(W.eval())
    #print()
    # post training loss
    print("Post training MSE: ", sess.run(loss, feed_dict={X: x_test, Y: y_test}))

    print("Actual sum: ", y_test[0:10])
    print()
    y_hat = sess.run(y_pred, feed_dict={X: x_test, Y: y_test})
    print("Predicted sum: ", y_hat[0:10] )