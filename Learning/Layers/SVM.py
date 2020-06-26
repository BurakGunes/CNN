# -*- coding: utf-8 -*-
"""
This file implements the well-known Support Vector Machine Algorithm.
"""

import numpy as np

class SVM:
    
    """SVM algorithm which can be run under the framework of the Sequential
    class. """
    
    def __init__(self, input_size, output_size, initialization_technique, C, N):
        
        """
        Initializes the SVM class and holds class-wide parameters.
        
        Params:
            input_size (int16) : Feature vector size
            output_size (int16) : For binary classification, this is 1 (-1,+1)
            C : regularization parameter
        """
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.initialization_technique = initialization_technique
        
        self.W = 0
        self.data_count = N
        self.C = C
        
        self.vx = 0
        
        
    def compile_model(self, input_size, output_size):
        
        """Although SVM class is usually used alone, this class is designed
        such that it can be between the layers of a larger network. Hence,
        it accepts the input size and output size before prooceeding any
        further.
        
        Params:
            input_size () : input size as a tuple
            output_size () : output size as a tuple. At the moment, only binary
            SVM case is implemented. Hence, the default output size is (2,1)
            no matter the argument.
            C () : Larger the C, more punished the algorithm for a mistakenly 
            classified sample. 
        
        Return:
            True (bool) : Always returns True.
        
        """
        
        self.input_size = input_size
        self.output_size = output_size
        
        self._initialize_weights()
        
        return True
    
    
    def _initialize_weights(self):
        
        """Given the initialization technique, this method randomly assigns
        numbers to weights present in the algorithm.
        
        Return:
            True (bool) : Always returns True
        
        """
        total_no_of_input = self.input_size
        self.W = np.random.randn(total_no_of_input, self.output_size)
        
        if self.initialization_technique == "He Normal":
            self.W /= np.sqrt(total_no_of_input / 2)
        elif self.initialization_technique == "Xavier":
            self.W /= np.sqrt(total_no_of_input)

        return True

        
    
    def forward(self, x):
        
        """Initiates a forward propagation given a single sample based on the
        current weights.
        
        Param:
            x (float64) : A numpy array of input features.
            
        Return:
            y_hat (float64) : Output of the algorithm.
        
        """
        
        y_hat = np.dot(self.W.T, x)
        
        return y_hat
    
    def backward(self, dy_hat, y):
        
        """ Taken the gradient of the cost as dy_hat and the corresponding
        "true" labels as y. Calculates weight changes which must be induced 
        on the existing ones.
        
        Params:
            dy_hat (float64) : dL/dy_hat
            y (int16) : correct labels.
            
        Return:
            dw (float64) : dL/dy_hat * dy_hat/da * da/dw, where a = phi(x*w)
        """
        
        dw = dy_hat #due to the definition of soft margin loss
        return dw, 0
        
        
    def sgd_momentum(self,
                     dw,
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
        
        self.vx = rho * self.vx + dw
        self.set_weights(learning_rate * self.vx)
        
        return True
      
        
    def set_weights(self, dw):
        
        """Updates the weights, i.e. :  
                w := w - weight_gradient
        Param:
            weight_gradient (float64) : the weight gradient which is supposed
            to be subtracted from the original weight vectors.
        """
        
        self.W -= dw
        
        return True
    
    def get_output_size(self):
        
        """Output size of fully connected layer is determined by the designer.
        For this reason, it must be made accessible to the Sequential class.
        
        Return:
            output_size (float64) : the output size of the final layer
            i.e. [10,1].
        """
        
        return self.output_size
            
        
    def save_weights(self, N):
        
        """This function takes the epoch number N and saves the associated
        weights which are present at the exact epoch N to the disk.
        
        Param:
            N (int16) : Epoch number which should be >1 """
            
        filename = "SVM_Epoch_" + str(N) + ".npy"
        np.save(filename, self.W)
        return True
    
    def load_weights(self, N):
        
        """After a training, all weights associated with epochs are saved. 
        This function lets the engineer choose the best performing weights
        and reload them into the memory from the disk.
        
        Param:
            N (int16): Epoch number
        """
        
        filename = "SVM_Epoch_" + str(N) + ".npy"
        self.W = np.load(filename)
        return True    
    
    
    def get_input_size(self):
        
        """Returns the associated input size as a tuple."""
        
        return self.input_size
    
    
    def predict(self, x):
        
        """Predict the output via the last weight vector trained by the class.
        
        Params:
            U (float64) : The test set in the form of pandas DataFrame. Don't
            forget to include the bias term.
            
        Returns:
            predictions (float64) : Predictions using the classes weight vector
            in the form of 1's and -1's.
        """
        prediction = np.sign(np.dot(x, self.W))
        return prediction
    