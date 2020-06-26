# -*- coding: utf-8 -*-
"""
Generate a convolution layer which is of configurable filter size.
"""

import numpy as np

from data_util.util import im2col

class Conv2D():
    
    def __init__(self,
                 filter_size = (3,3,8),
                 input_size = None,
                 initialization_technique = "Xavier",
                 nonlinearity = "ReLU"):

        
        """Initialize a convolutional layer with the given parameters. Filters
        are square matrices. 
        Params:
            input_size (uint8) : size of the input matrix, i.e. given an [MxM]
            matrix, size is (M,M) tuple.
            *Only specify if it is the input layer*.
            
        """
        
        self.filter_size = filter_size
        self.input_size = input_size
        #define and store last the output and input
        self.output = np.array(())
        self.input = np.array(())
        #parameter models:
        self.initialization_technique = initialization_technique
        self.vx = 0 #Momentum term to be initialized later.
        self.nonlinearity_fun = nonlinearity
        
    
    def compile_model(self, input_size, output_size):
        
        """This class is designed such that it can handle only fixed size in-
        puts. For this reason, before going **forward**, model parameters 
        should be fixed with respect to surrounding layers. 
        """
        
        self.output_size = output_size
        self.input_size = input_size
        self._initialize_output()
        self._initialize_weights()
    
    def _initialize_weights(self):
        
        """Initialize the weights of the filters with the given dimensions of
        surrounding layers. 
        """
        
        #according to the Stanford's CS231n definition. (lecture 6)
        fan_in = self.filter_size[0] * self.filter_size[1] * self.filter_size[2]
        
        #zero mean gaussian...
        self.W = np.random.randn(self.filter_size[0],
                                  self.filter_size[1],
                                  self.filter_size[2])

        
        if self.initialization_technique == "He Normal":
            self.W /= np.sqrt(fan_in / 2)
        elif self.initialization_technique == "Xavier":
            self.W /= np.sqrt(fan_in)
        else:
            return False #indicating that weights are not properly initialized.
        
        return True
        
    def _initialize_output(self):
        
        """Initialize the output given the parameters of the convolutional
        layer. Should not be called from outside. Only sequential layer 
        should handle the output size determination function.
        
        Param:
            output_size (uint8) : output size of the layer, e.g. a tuple of 
            (26, 26, 8). 
            
        Return:
            (Bool) : A successful operation returns True. 
        """
        
        self.output = np.zeros(self.output_size)
        return True

    def forward(self, input):
        
        """*Valid* Forward propagation algorithm. Implemented using im2col 
        algorithm which allows for great optimizationsin terms of performance 
        parameters.
        
        Stride = 1
        Padding = 0
        by default. 
        
        Param:
            input (float64) : input term received from the preceding layer.
        Return:
            output (float64) : Output product of the input and filters. 
        """
        
        #save the last valid input for backprop purposes.
        self.input = input
        #convert the input into a column vector.
        input_col = im2col(input.T, self.filter_size[0:2])
        #convert the filters into column vectors.
        W_col = self.W.reshape(self.filter_size[0]*self.filter_size[1], -1)
        #apply the convolution. 
        out = np.matmul(input_col.T, W_col)
        #return back to the original shape. 
        
        out = self.nonlinearity(out)
        
        self.output = out.reshape(self.output_size)
        
        return self.output
    
    def backward(self, error):
        
        """Implements the backpropagation algorithm. Takes the error and ...
        
        Param:
            error (float64) : error term received from the proceeding layer.
        
        Returns:
            deltaL_delta_w (float64) : weight changes to induce on the existing
            ones.
            deltaL_delta_x (float64) : unfortunately, I couldn't  find an 
            *efficient* way of implementing it. 
        
        """
        
        #initialize the loss gradient w.r.t. the weights (i.e. the filter)
        deltaL_delta_w = np.zeros(self.filter_size)
        #convert the error into columns
        block_size = (self.input_size[1]-self.filter_size[1]+1,
                      self.input_size[0]-self.filter_size[0]+1)
        
        #nonlinearity
        # inp = self.input
        inp = self.nonlinearity(self.input, gradient = True)
        
        #convert the last input into the desired column vector form.        
        input_col = im2col(inp.T, block_size)
        #initialize the column converted version of error, i.e. dL/dY
        error_cols = np.zeros((self.output_size[0] * self.output_size[1],
                              self.output_size[2]))
        
        
        for i in np.arange(0,  self.filter_size[2]):
            col = error[:,:,i]
            error_cols[:,i] = col.flatten()
            
        #initialize the dL/dW matrix in the form of column vectors
        deltaL_delta_w_cols = np.zeros((self.filter_size[0] * self.filter_size[1],
                                        self.filter_size[2]))
        deltaL_delta_w_cols = np.matmul(input_col.T, error_cols)
        deltaL_delta_w = deltaL_delta_w_cols.reshape(self.filter_size)
        
        return deltaL_delta_w, 0

    
    
    def sgd_momentum(self,
                     weight_grad,
                     learning_rate = 0.000001,
                     rho = 0.90):
        
        """Implements the stochastic gradient descent with momentum algorithm.
        Basically, this algorithm builds up a velocity of gradient. The jittery
        behaviour while finding extremums goes away as this algorithm has that
        action that averages out the noise. (Stanford Lecture 7)
        
        Params:
            layer () : layer to update its weights
            learning rate (float64) : a hyperparameter
            weight_grad (float64) : the output calculated from the backprop.
            rho (float64) : another hyperparameter which should be set.
        """
        
        self.vx = rho * self.vx + weight_grad
        self.set_weights(learning_rate * self.vx)
    
    def set_weights(self, weight_gradient):
        
        """Updates the weights, i.e. :  
                w := w - weight_gradient
        Param:
            weight_gradient (float64) : the weight gradient which is supposed
            to be subtracted from the original weight vectors.
        """
        
        self.W -= weight_gradient

    
    def get_input_size(self):
        
        """"""
        
        return self.input_size

    def get_filter_size(self):
        
        """"""
        
        return self.filter_size
    
    def get_output_size(self):
        
        """"""
        
        return self.output.shape
    
    
    def save_weights(self, N):
        
        """This function takes the epoch number N and saves the associated
        weights which are present at the exact epoch N to the disk.
        
        Param:
            N (int16) : Epoch number which should be >1 """
            
        filename = "Conv2D_Epoch_" + str(N) + ".npy"
        np.save(filename, self.W)
        return True

    def load_weights(self, N):
        
        """After a training, all weights associated with epochs are saved. 
        This function lets the engineer choose the best performing weights
        and reload them into the memory from the disk.
        
        Param:
            N (int16): Epoch number
        """
        
        filename = "Conv2D_Epoch_" + str(N) + ".npy"
        self.W = np.load(filename)
        return True
    
    
    def nonlinearity(self, x, gradient = False):
        
        """Add a nonlinearity to the pooling layer. At the moment, only ReLU is
        supported...
        
        Taken from "stackoverflow" but dont remember the exact place.
        
        Param:
            x (float64) : usually the output of the layer.
        return:
            x_out (float64) : output of the chosen nonlinear function.
        """
        
        if self.nonlinearity_fun == "ReLU":
            if gradient:
                return 1 * (x > 0)
            else:
                return x * (x > 0)
        elif self.nonlinearity_fun == "Linear":
            return x
        else:
            raise Exception("The given nonlinearity function is not yet implemented.")
