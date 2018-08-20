import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

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
                            trainable=True, name="W_hat")

    M_hat = tf.get_variable(shape=[in_shape, out_units],
                            initializer=tf.initializers.random_uniform(minval=-2, maxval=2),
                            trainable=True, name="M_hat")

    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    # Express Input feature in log space to learn complex functions
    x_modified = tf.log(tf.abs(x_in) + epsilon)

    m = tf.exp( tf.matmul(x_modified, W) )

    return m, W

# Test the Network by learning the adition

# Generate a series of input number X1 and X2 for training
x1 = np.arange(0,10000,5, dtype=np.float32)
x2 = np.arange(5,10005,5, dtype=np.float32)


y_train = x1 * x2

x_train = np.column_stack((x1,x2))

print(x_train.shape)
print(y_train.shape)

# Generate a series of input number X1 and X2 for testing
x1 = np.arange(1000,2000,8, dtype=np.float32)
x2 = np.arange(1000,1500,4, dtype= np.float32)

x_test = np.column_stack((x1,x2))
y_test = x1 * x2

print()
print(x_test.shape)
print(y_test.shape)


# Define the placeholder to feed the value at run time
X = tf.placeholder(dtype=tf.float32, shape =[None , 2])    # Number of samples x Number of features (number of inputs to be added)
Y = tf.placeholder(dtype=tf.float32, shape=[None,])

# define the network
# Here the network contains only one NAC cell (for testing)
y_pred, W = nac_complex_single_layer(X, out_units=1)
y_pred = tf.squeeze(y_pred)             # Remove extra dimensions if any

# Mean Square Error (MSE)
loss = tf.reduce_mean( (y_pred - Y) **2)
#loss= tf.losses.mean_squared_error(labels=y_train, predictions=y_pred)



# training parameters
alpha = 0.5    # learning rate
epochs = 36000

optimize = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

with tf.Session() as sess:

    #init = tf.global_variables_initializer()
    cost_history = []

    sess.run(tf.global_variables_initializer())

    # pre training evaluate
    print("Pre training MSE: ", sess.run (loss, feed_dict={X: x_test, Y:y_test}))
    print()
    for i in range(epochs):
        _, cost = sess.run([optimize, loss ], feed_dict={X:x_train, Y: y_train})
        print("epoch: {}, MSE: {}".format( i,cost) )
        cost_history.append(cost)

    # plot the MSE over each iteration
    plt.plot(np.arange(epochs),np.log(cost_history))  # Plot MSE on log scale
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()

    print()
    print(W.eval())
    print()
    # post training loss
    print("Post training MSE: ", sess.run(loss, feed_dict={X: x_test, Y: y_test}))

    print("Actual sum: ", y_test[0:10])
    print()
    print("Predicted sum: ", sess.run(y_pred[0:10], feed_dict={X: x_test, Y: y_test}))