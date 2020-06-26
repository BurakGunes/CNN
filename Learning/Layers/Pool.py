# -*- coding: utf-8 -*-
"""
Pooling layer which can be configured to be maximum pooling, averaging, etc.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


class Pool():
    
    def __init__(self, 
                 poolsize = (2,2),
                 technique = "Max"):
        
        """Initialize the pooling layer with the appropriate parameters. 
        
        Params:
            poolsize (uint8) : Size of the pooling operator which should be in the
            form of a tuple, i.e. (2x2), (3x3)
            technique (String) : max, mean, and some other techniques are to be
            supported. At the moment, only mean is supported.
        """
        
        #dimensional params
        self.poolsize = poolsize
        self.output_size = np.array(())
        self.input_size = np.array(())
        self.output = np.array(())
        self.input_size = np.array(())
        #store the last input and output
        self.output = np.array(())
        self.input = np.array(()) #store the input as it enters into the layer
        self.max_positions = np.array(())
        #technique related params.
        self.technique = technique
        
    def compile_model(self, input_size, output_size):
        
        """Finalizes the parameter determination of the pooling layer.
        
        Params:
            input_size (uint8) : the size of the preceeding layer, a tuple ().
            output_size (uint8) : the size of the proceeding layer, a tuple ().
        Return:
            (Bool) : True when everything goes well.
        """
        
        self.input_size = input_size
        self.output_size = output_size
        self._initialize_output()
        self.max_positions = np.zeros(shape = self.input_size, dtype = bool)
                
        return True
    
    def _initialize_output(self):
        
        """Initializa the output parameters given parameter. This method should
        be called by a Sequential class instance only at the beginning while
        initializing a CNN. This method determines the shape of the output.
        
        Param:
            output_size (uint8) : the output size in the form of a tuple
            
        Return:
            (Bool) : True if all goes well.
        """
        
        self.output = np.zeros(self.output_size)
        
        return True
                    
    def forward(self, input):
        
        """Given the input, this method implements the pooling algorithm with 
        the preferred technique.
        
        Params:
            input (float64): An [MxMxN] numpy matrix where M is the output width 
            that came out from the convolution operation. N is the numbers of
            filters used in the convolution layer.
            
        Returns:
            out (float64) : The output of the pooling layer inthe form of [MxMxN]
        """
        
        
        self.input = input
        out = np.zeros(self.output_size)
        for i in np.arange(0, self.output_size[2]):
            out[:,:,i] = self.pool2d(input[:,:,i],
                                     block_size = self.poolsize[0],
                                     stride = self.poolsize[0])
        
        self.output = out
        return self.output
    
    def backward(self, error):
        
        """Given the output of the layer (which is often the dLoss/d(input)of
        the proceeding layer, i.e. error), this function implements the 
        backpropagation algorithm. In other words, takes the error from its back end and places
        gradient values to a matrix in the form of the original input to the 
        layer and places the error values in such a way that pooling function 
        is reversed. #note to myself: check the description of error. 
        
        Param:
            error (float64) : error from the proceeding layer.
        """
        err = error / (self.poolsize[0] * self.poolsize[1])
        deltaL_delta_x = np.repeat(np.repeat(err, self.poolsize[0], axis = 0), self.poolsize[1], axis = 1)
        
        # max_pos = self.max_positions.flatten()
        # deltaL_delta_x[max_pos == True] = err
        
        # deltaL_delta_x = np.reshape(deltaL_delta_x, self.input_size)
                            
        return deltaL_delta_x
    
    
    def pool2d(self, input, block_size, stride):
        """
        *Not my own code*. As the pooling operation itself is not directly related
        with out own coursework, I have taken the liberty of taking the code 
        from elsewhere. My own original implementation was much, much slower.
        
        Source:
        https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy
        """

        output_shape = ((input.shape[0] - block_size)//stride + 1,
                        (input.shape[1] - block_size)//stride + 1)
        kernel_size = (block_size, block_size)
        input_w = as_strided(input,
                         shape = output_shape + kernel_size,
                         strides = (stride*input.strides[0], stride*input.strides[1]) + input.strides)
        
        input_w = input_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        return input_w.mean(axis=(1,2)).reshape(output_shape)

    def save_weights(self, N):
        
        """Pooling layer has no trainable weights. Therefore, this function 
        always returns true."""
            
        return True
    
    def load_weights(self, N):
        
        """Pooling layer has no trainable weights. Therefore, this function 
        always returns true."""
            
        return True
    
