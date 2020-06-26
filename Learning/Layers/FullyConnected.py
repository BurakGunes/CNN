# -*- coding: utf-8 -*-
"""A fully connected neural network with one hidden layer.
"""

import numpy as np

class FullyConnected():
    
    
    def __init__(self,
                 activation = "SoftMax",
                 output_size = 2,
                 input_size = None, #leave it if this is not the first layer.
                 initialization_technique = "Xavier"):
        
        """Initialize a fully connected layer of neural network. 
        """
        
        self.W = np.array(())
        self.activation = activation
        self.output_size = output_size
        self.input_size = 0
        self.initialization_technique = initialization_technique
        #store the last input and corresponding output value
        self.y_hat = np.array(())
        self.x = np.array(())
        self.a = np.array(()) #a is the value of activation, i.e. x*W = a,
        #velocity related with the sgd+momentum algorithm
        self.vx = 0
        #log epsilon param
        self.eps = 1e-10
    
    
    def compile_model(self, input_size, output_size):
        
        """Model parameters with respect to the preceeding layer must be set
        before running the model.
        """
        
        self.input_size = input_size
        #do nothing with the output size
        self._initialize_weights()
        
        return True
        
        
    def _initialize_weights(self):
        
        """Initialize the weights of the filters with the given dimensions of
        surrounding layers. 
        """
        total_no_of_input = self.input_size[0] * self.input_size[1] * self.input_size[2]
        self.W = np.random.randn(total_no_of_input, self.output_size)
        
        if self.initialization_technique == "He Normal":
            self.W /= np.sqrt(total_no_of_input / 2)
        elif self.initialization_technique == "Xavier":
            self.W /= np.sqrt(total_no_of_input)
            
            
    def forward(self, x):
        
        """Implements a forward propagation algorithm. Only for softmax at the
        moment.
        
        Params:
            x (float64) : an [NxN] input numpy matrix.
            
        Return:
            out (float64) : output predictions out of the final layer.
        """
        
        #*assume* that the input is not flat and store it as a vector.
        self.x = np.reshape(x.flatten(), (-1,1))
        self.a = np.matmul(self.x.T, self.W).T #a is the value inside the activation func.
        
        if self.activation == "SoftMax":
            self.y_hat = self.softmax()
            
        return self.y_hat
    
    
    def backward(self, dy_hat, y):
        
        """Takes an error gradient dy_hat to calculate the loss gradient
        with respect to the weights of the layer as well as to the input of the
        layer.
        
        Param:
            dy_hat (float64) : the error gradient which must be in the form of
            the output of the layer.
            
        Returns:
            dw (float64) : the error with respect to weights.
            dx (float64) : to be used in the previous layer.
        """
        
        #save the correct label associated with this pass.
        self.y = y
        
        #dw = dL/dy_hat * dy_hat/da * da/dw, L is the loss function
        #a = xW, da/dw = x
        
        deltaL_delta_y_hat = dy_hat
        delta_y_hat_delta_a = self.calculate_delta_y_hat_delta_a()#size=(10,1) 
        #a quick fix...
        deltaL_delta_a = delta_y_hat_delta_a * deltaL_delta_y_hat         
        delta_a_delta_w = self.x
        
        #deltaL_delta_a = np.dot(deltaL_delta_y_hat.T, delta_y_hat_delta_a)
        deltaL_delta_w = np.matmul(delta_a_delta_w, deltaL_delta_a.T)
        
        dw = deltaL_delta_w
                
        delta_a_delta_x = self.W
        deltaL_delta_x = np.matmul(delta_a_delta_x, deltaL_delta_a)
        dx = np.reshape(deltaL_delta_x, self.input_size)
        
        return dw, dx

 
    def softmax(self, gradient = False):
        
        """Implements the softmax function and its gradient with respect to
        the activations, i.e. a = xW.
        
        Param:
            gradient (Bool) : If true, this function calculates the gradient.
        Return:
            dy_hat_d_a (float64) : The gradient of the softmax function w.r.t.
            the activations in the form of a vector.
            sm (float64) : Softmax function applied to the output of the layer
            in the form of a vector.
        """
            
        if gradient:#w.r.t. activations
            cl = np.argmax(self.y) #Get the position of correct label. 
            sm = self.softmax() #Get the probability distribution vector. 
            sc = sm[cl] #get the prob. dist. assoc. with the correct class.
            dy_hat_d_a = -sm * sc
            dy_hat_d_a[cl] = sc * (1 - sc)
            
            return np.reshape(dy_hat_d_a, (-1, 1))
        else:
            num = np.exp(self.a)
            denum = np.sum(num)
            sm = num / denum
            return sm
    
    
    def calculate_delta_y_hat_delta_a(self):
        
        """Calculates the dy_hat/da where a = xW and y_hat = g(a). The function
        g is the specified activation function associated with the layer. For
        example ReLU or SoftMax if it is the final layer.
        
        Return:
            delta_y_hat_delta_a (float64) : the gradient of the output w.r.t.
            the activations in the form of a vector (i.e. a column vector of
            form (10,1))
        """
        
        delta_y_hat_delta_a = np.array(())#initialize
        
        if self.activation == "SoftMax":
            delta_y_hat_delta_a = self.softmax(gradient = True)
        elif self.activation == "ReLU":
            delta_y_hat_delta_a = None #not supported at the moment.
        else:
            delta_y_hat_delta_a = None
        
        return delta_y_hat_delta_a
        
            
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
      
        
    def set_weights(self, dw):
        
        """Updates the weights, i.e. :  
                w := w - weight_gradient
        Param:
            weight_gradient (float64) : the weight gradient which is supposed
            to be subtracted from the original weight vectors.
        """
        
        self.W -= dw
    
    
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
            
        filename = "FC_Epoch_" + str(N) + ".npy"
        np.save(filename, self.W)
        return True
    
    def load_weights(self, N):
        
        """After a training, all weights associated with epochs are saved. 
        This function lets the engineer choose the best performing weights
        and reload them into the memory from the disk.
        
        Param:
            N (int16): Epoch number
        """
        
        filename = "FC_Epoch_" + str(N) + ".npy"
        self.W = np.load(filename)
        return True    