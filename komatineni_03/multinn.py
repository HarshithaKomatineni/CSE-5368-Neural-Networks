# Harshitha , Komatineni
# 1001968082
# 2020_10_30
# Assignment_03_01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimensions = input_dimension
        self.W = list()
        self.B = list()
        self.Begin_layer = list()
        self.node_count= list()

    
    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """

        if(len(self.node_count)==0):    
            self.W.append(tf.Variable(np.random.randn(self.input_dimensions,num_nodes),name="W"+str(len(self.node_count)),trainable=True))
        else:

            self.W.append(tf.Variable(np.random.randn(self.node_count[-1],num_nodes),name="W"+str(len(self.node_count)),trainable=True))
        

        self.B.append(tf.Variable(np.random.randn(1,num_nodes),name="B"+str(len(self.node_count)),trainable=True))
        self.Begin_layer.append(transfer_function.lower())
        self.node_count.append(num_nodes)
        return
        
        

    def get_weights_without_biases(self, layer_number):
        """matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
         
         
        return self.W[layer_number].numpy()
         
    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
         
        return self.B[layer_number].numpy()

    def set_weights_without_biases(self, W, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param W: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
         
        if(W.shape == self.W[layer_number].numpy().shape):
            self.W[layer_number].assign(W) 
        else:
            return -1
         

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        
        if(biases.shape == self.B[layer_number].numpy().shape):
            self.B[layer_number].assign(biases) 
        else:
            return -1

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        
        
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """

        Y = tf.Variable(X,trainable=True)
        for i in range(len(self.node_count)):
            if(self.Begin_layer[i]=="sigmoid"):
                Y = tf.nn.sigmoid(tf.matmul(Y,self.W[i]) + self.B[i])
            elif(self.Begin_layer[i]=="relu"):
                Y = tf.nn.relu(tf.matmul(Y,self.W[i]) + self.B[i])
            else:
                Y = tf.matmul(Y,self.W[i]) + self.B[i]
        return Y        

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
         
        for i in range(num_epochs):
            for j in range(0,X_train.shape[0],batch_size):
                if((j+batch_size)<=X_train.shape[0]):
                        with tf.GradientTape(persistent = True) as tape:
                            y_pred = self.predict(X_train[j:j+batch_size,:])
                            loss = self.calculate_loss(y_train[j:j+batch_size],y_pred)
                        for k in range(len(self.node_count)-1,-1,-1):
                            dloss_dw, dloss_db = tape.gradient(loss, [self.W[k], self.B[k]])
                            self.W[k].assign_sub(alpha * dloss_dw)
                            self.B[k].assign_sub(alpha * dloss_db) 

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        
        y_pred1 = self.predict(X).numpy()
        y_pred = np.zeros((y_pred1.shape[0],1),dtype='int32')
        count=0
        for i in range(y_pred1.shape[0]):
            y_pred[i] = np.argmax(y_pred1[i])
            if(y[i]!=y_pred[i][0]):
                count+=1
        return count/X.shape[0]

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        y_pred1 = self.predict(X).numpy()
        y_pred = np.zeros((y_pred1.shape[0],1),dtype='int32')
        for i in range(y_pred1.shape[0]):
            y_pred[i] = np.argmax(y_pred1[i])
        return tf.math.confusion_matrix(y,y_pred,num_classes=self.node_count[-1])