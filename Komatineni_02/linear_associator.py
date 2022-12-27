# Komatineni , Harshitha
# 1001968082
# 2022_10_09
# Assignment_02_01

import numpy as np


class LinearAssociator(object):

    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """

        self.input_dimensions =input_dimensions;
        self.number_of_nodes = number_of_nodes;
        self.transfer_function = transfer_function;
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """

        np.random.seed(seed);

        self.internalweights =np.zeros((self.number_of_nodes , self.input_dimensions))
        self.internalweights = np.random.randn(self.number_of_nodes , self.input_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        self.internalweights = W
    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        weightmatrix = self.internalweights;
        return weightmatrix
    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """

        if(self.transfer_function == "Linear"):
            output = np.dot(self.internalweights, X)
            npoutput = np.array(output)
        else:
            output = np.dot(self.internalweights, X)
            npoutput = np.array(output)
            npoutput[npoutput<=0]=0
            npoutput[npoutput>0] = 1
        return npoutput
    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        invX = np.linalg.pinv(X);
        self.internalweights = (np.dot((np.linalg.pinv(X)).T,y.T)).T;

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        finalval =X.shape[1]
        for k in range(num_epochs):
            for j in range(0, finalval, batch_size):
                x = X[:, j:j + batch_size]
                Y = y[:, j:j + batch_size]
                predictedvalue = self.predict(x)
                if learning == 'Filt':
                    self.internalweights = (1 - gamma) * self.internalweights + alpha * (Y.dot(x.T))
                elif learning == 'Delta' or learning == 'delta':
                    self.internalweights =  ((Y - predictedvalue).dot(x.T))*alpha + self.internalweights
                elif learning == 'Hebb':
                    self.internalweights = self.internalweights + alpha * (predictedvalue.dot(x.T))

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        predictedvalue = self.predict(X)
        error = predictedvalue-y;
        squareerror = np.square(error)
        squaresum=np.sum(squareerror)
        sizesquareerror = np.size(squareerror);
        meansquareerror = squaresum/sizesquareerror;

        return meansquareerror