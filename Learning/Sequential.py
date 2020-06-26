# -*- coding: utf-8 -*-
"""
Sequential convolutional neural network building class that lets you bring 
together various layers such as convolution, pooling, fully connected, etc.
"""

import numpy as np

import matplotlib.pyplot as plt
import time

from Learning.Layers.Conv2D import Conv2D
from Learning.Layers.Pool import Pool
from Learning.Layers.FullyConnected import FullyConnected
from Learning.Layers.SVM import SVM

class Sequential():

    def __init__(self):
        self.layers = []
        self.layer_count = 0
        
        #store output, input and correct label
        self.x = np.array(())
        self.y_hat = np.array(())
        self.y = np.array(())
        
        #inside log, an epsilon param is needed to prevent it the log from being log0.
        self.eps = 1e-5 
        
        #optimization params:
        self.learning_rate = 0
        self.rho = 0
        
        #termination param:
        self.auto_terminate_loss = 0.0000000000000001
        
        #store average losses per epoch both for the training and validation. 
        self.train_losses = []
        self.valid_losses = []
    
    def add(self, layer):
        
        """Add a layer to the convolutional neural network in a sequential 
        manner. Currently, the code is a mess but it works, seemingly. 
        
        Param:
            layer () : A layer instance
        """
        
        
        if self.layer_count == 0:
            self.layers.append(layer)
            calculated_output_size = self.compute_nextlayer_size(layer)
            layer.compile_model(layer.get_input_size(), calculated_output_size)
            self.next_layer_input_size = calculated_output_size
            self.layer_count += 1  
        elif isinstance(layer, FullyConnected):
            self.layers.append(layer)
            calculated_output_size = self.compute_nextlayer_size(layer)
            #nextlayerinputsize in the form of (AxAxB). The corresponding layer
            #must flatten it. 
            layer.compile_model(self.next_layer_input_size, calculated_output_size)
            self.next_layer_input_size = calculated_output_size
            self.layer_count += 1
        else:
            self.layers.append(layer)
            calculated_output_size = self.compute_nextlayer_size(layer)
            layer.compile_model(self.next_layer_input_size, calculated_output_size)
            self.next_layer_input_size = calculated_output_size
            self.layer_count += 1
            
        return True
               
    def compute_nextlayer_size(self, layer):
        if isinstance(layer, Conv2D):
            input_size = layer.get_input_size()
            filter_size = layer.get_filter_size()
            
            outputsize = (input_size[0] - filter_size[0] + 1,
                    input_size[1] - filter_size[1] + 1,
                    filter_size[2])
            return outputsize
        elif isinstance(layer, Pool):
            input_size = self.next_layer_input_size
            outputsize = (input_size[0] // layer.poolsize[0], input_size[1] // layer.poolsize[1], input_size[2])
            return outputsize
        elif isinstance(layer, FullyConnected):
            outputsize = layer.get_output_size()
            return outputsize
        elif isinstance(layer, SVM):
            outputsize = layer.get_output_size()
            return outputsize
        
    def calculate_loss(self, y, y_hat, gradient = False):
        
        """Calculates the loss for a single sample given the parameters."""
        
        if self.loss_fun == "CrossEntropy":
            return self.cross_entropy_loss(y, y_hat, gradient) 
        elif self.loss_fun == "SoftMargin":
            #at the moment, only the last layer can be the SVM layer.
            svm_layer = self.layers[-1] #get the layer which hosts the SVM layer.
            
            W = svm_layer.W
            N = svm_layer.data_count
            C = svm_layer.C
            
            x = self.last_input

            return self.soft_margin_loss(W, N, C, y, x, y_hat, gradient)
        
        return None

    
    def forward(self, x, y):
        
        """Implements a forward propagation algorithm over all the layers.
        Takes only one sample at a time. 
        
        Params:
            x (float64) : One sample of the input to be trained.
            y (float64) : One label of the input to be trained.
        Returns:
            y_hat (float64) : Prediction of the algorithm.
            y (float64) : The correct labels.
        """
        
        self.last_input = x
        self.last_label = y
        
        out = x
        for layer in self.layers:
            out = layer.forward(out)
                    
        loss = self.calculate_loss(y, out, gradient = False)
        return out, loss
             
    def backward(self, y, y_hat):
        
        """Implements a backward propagation (i.e. backpropogation) algorithm
        to train the network over the training set.
        
        Params:
            y (float64) : One label of the input to be trained.
        """
        
        #vanilla backprop for the FC layer, at the moment.
        
        #deltaL / delta y_hat
        dy_hat = self.calculate_loss(y, y_hat, gradient = True)
        
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                dw, dy_hat = layer.backward(dy_hat, y)
                layer.sgd_momentum(dw, self.learning_rate, self.rho)
            elif isinstance(layer, Pool):
                #pooling layer can't be trained, so just passing the gradients.
                dy_hat = layer.backward(dy_hat)
            elif isinstance(layer, SVM):
                dw = layer.backward(dy_hat)
                layer.sgd_momentum(dw, self.learning_rate, self.rho)
            else:
                #dw = deltaL / delta w
                dw, dy_hat = layer.backward(dy_hat)
                layer.sgd_momentum(dw, self.learning_rate, self.rho)
                
        return True

    def set_model_parameters(self,
                             learning_rate = 0.000001,
                             rho = 0.90,
                             optimizer = "SGD+Momentum",
                             loss = "CrossEntropy",
                             initial_epoch = 1):
        
        """Set the model parameters such as learning rate, rho in SGD+momentum,
        which are hyperparameters in the learning process. Also specify the 
        optimization algorithm."""
        
        self.learning_rate = learning_rate
        self.rho = rho
        self.loss_fun = loss
        self.initial_epoch = initial_epoch
        
        return True
    
    def add_epoch_losses(self, new_train_loss, new_val_loss):    
        
        """ Takes average losses both for the training and validation and 
        stores them in a list.
        
        Params:
            new_train_loss (float64) : Average loss after training for 
                                       one epoch
            new_val_loss (float64) : Average loss following completion of
                                     validation.
        Return:
            True (bool) 
        """
        
        self.train_losses.append(new_train_loss)
        self.valid_losses.append(new_val_loss)
        
        return True
    
    def plot_losses(self):
        
        """Plots the training and validation losses encountered so far."""
        
        epoch_count = len(self.train_losses)
        assert epoch_count == len(self.valid_losses) 
        
        epochs = np.arange(self.initial_epoch, epoch_count+ self.initial_epoch) 
        
        plt.plot(epochs, self.train_losses, "-g", label = "Training losses")
        plt.plot(epochs, self.valid_losses, "-r", label = "Validation losses")
        plt.legend(loc = "upper right")
        plt.xlabel("Epochs")
        plt.ylabel("Average loss per epoch")
        plt.title("Losses w.r.t. Advancing Epoch Number")
        plt.show()
        
        return True     
    
    def fit_model(self, X_train, Y_train,
                  X_valid = None, Y_valid = None,
                  epoch = 5, verbose = False,
                  perEpochVerbose = 1, weightSavePerEpoch = 1):
        
        """*SGD Only*
        This method tries to calculate an underlying function which can
        fit to the underlying distribution of the training data. Validation
        is to be implemented later. """
        
        def autoterminate(loss):
            return True if loss < self.auto_terminate_loss else False
        
        
        beginning_time = time.time()
        pmt = np.random.permutation(X_train.shape[0]) #randomize the input vector
        
        print("Training started...")
        
        for e in range(epoch):
            epoch_begin = time.time()
            pmt = np.random.permutation(X_train.shape[0])
            
            #training
            epoch_loss = 0; total_epoch_loss = 0
            for i in pmt:
                xx = X_train[i,:,:] if X_train.ndim == 3 else X_train[i,:]
                yy = Y_train[i,:] if Y_train.ndim == 2 else Y_train[i]
                out, loss = self.forward(xx, yy)
                self.backward(yy, out) 
                total_epoch_loss += loss
            epoch_loss = float(total_epoch_loss / X_train.shape[0])
            
            #validation...
            val_set_len = X_valid.shape[0]
            total_val_loss = 0
            for i in range(val_set_len):
                xx = X_train[i,:,:] if X_train.ndim == 3 else X_train[i,:]
                yy = Y_train[i,:] if Y_train.ndim == 2 else Y_train[i]
                _, loss = self.forward(xx, yy)
                total_val_loss += loss
            val_loss = float(total_val_loss / val_set_len)
            
            #record the performance and plot it.
            self.add_epoch_losses(epoch_loss, val_loss)
            
            if e % perEpochVerbose == 0:
                self.plot_losses()
            
            #save the weights of trainable layers...
            if e % weightSavePerEpoch == 0:
                self.save_weights(e+self.initial_epoch)
            
            #Calculate the time spent on this epoch.
            epoch_finish = time.time()  
            epoch_time = epoch_finish - epoch_begin
            
            if verbose:
                if e % perEpochVerbose == 0:
                    print(f"Epoch {e+1+self.initial_epoch}/{epoch+self.initial_epoch}: average loss: {epoch_loss}")
                    print(f"Time elapsed in one epoch for {X_train.shape[0]} samples is {round(epoch_time, 2)} s.")
                        
            if autoterminate(epoch_loss):
                print("Learning process is stopped with autotermination.")
                break #stop the learning process if the loss incurred is small.
            
            
        end_time = time.time()
        total_time_elapsed = end_time - beginning_time
        
        if verbose:
            print(f"Total time elapsed for {epoch} epoch is {round(total_time_elapsed, 2)} s.")
        
    def soft_margin_loss(self, W, N, C, y, x, y_hat, gradient = False):
        """Calculates the soft margin loss for the binary SVM classifier.
        
        Params:
            W (float64) : weights of the SVM classifier
            N (float64) : total number of samples used to train the SVM
            C (float64) : defines how much the SVM must be punished for a 
            wrongly classified sample.
            y (float64) : label of the last input to the system
            x (float64) : last input to the SVM
            gradient (bool) : if True, gradient of the loss is returned.
        Return:
            loss (float64) : Depending on the gradient parameter,
            loss is calculated.
        """
        
        if gradient:
            d = max(1 - y * y_hat, 0)
            return W / N if d == 0 else (W - C * y * x.reshape(-1,1)) / N
        else:
            loss = (0.5 * np.dot(W.T, W) + C * max(0, 1 - y * y_hat)) / N
            return loss
        
        return None
   
    
    def cross_entropy_loss(self, y, y_hat, gradient = False):
        
        """Calculates the cross entropy cost for multi class classification for
        the case of a single sample.
        
        Params:
            y (float64) : The correct labels
            y_hat (float64) : The output from the network.
            gradient (Bool) : If true, calculates the gradient. 
            
        Return:
            loss (float64) : loss with the cross entropy definition.
            dy (float64) : loss gradient with respect to the output.
        """
        
        #a crude way of computing a vector. change this later.
        
        yj = np.dot(y.T, y_hat) + self.eps
        
        if gradient:
            dy = -1 / yj
            return dy
        else:
            loss =-np.log(yj)
            return loss
        
        return None
        
    
    def accuracy(self, X, Y):
        
        """Calculates the accuracy given the samples and corresponding labels.
        
        Params:
            X (float64) : A numpy array of the samples.
            Y (float64) : A numpy array of the labels.
            
        Return:
            accuracy (float64) : correct guesses / sample count
        """
        
        correct_guesses = 0
        sample_count = Y.shape[0]
        
        for i in range(sample_count):
            guess, _ = self.forward(X[i, :, :], Y[i, :])
            guessed_label = np.argmax(guess)
            correct_label = np.argmax(Y[i, :])
            if guessed_label == correct_label:
                correct_guesses += 1
            
        accuracy = correct_guesses / sample_count
        return accuracy
    
    def save_weights(self, N):
        
        """This function initiates a saving procedure for all the layers
        present in the Sequential class. After completing a training process,
        the best performing epoch's weights can be reused.
        
        Param:
            N (int16): Epoch number
        """
        
        for layer in self.layers:
            layer.save_weights(N)
        
        return True
      
    def load_weights(self, N):
        
        """After a training, all weights associated with epochs are saved. 
        This function lets the engineer choose the best performing weights
        and reload them into the memory from the disk.
        
        Param:
            N (int16): Epoch number
        """
        
        for layer in self.layers:
            layer.load_weights(N)
            
        return True
    
    
    def predict(self, x, y):
        
        """Given individual samples x and y, it returns the algorithm's
        prediction."""
        
        y_hat, _ = self.forward(x, y)
        # (1,0) label stands for preictal
        # (0,1) label stands for interictal (minority)
        
        if isinstance(self.layers[-1], SVM):
            prediction = self.layers[-1].predict(x)
            if prediction == 1:
                return (1, 0)
            else:
                return (0, 1)
        else: #for CNN
            if np.argmax(y_hat) == 0:
                return (1, 0)
            else:
                return (0, 1)

        
    
    def test(self, x, y, verbose = False):
        
        """Given a test set, this function calculates the trained network's
        performance parameters. For this specific project, the parameters below
        are calculated:
            
            accuracy =
            ((# of correctly classified preictal) + (# of c. c. interictal)) /
            (number of total samples)
            
            specivity =
            (# of correctly classified interictal) /
            ((# samples mistaken for preictal) + (correctly classified interictal)) 
            
            sensitivity =
            (correctly classified preictal) /
            ((correctly classified preictal) + (mistaken for interictal))
            
        Params:
            x (float64) : test data
            y (float64) : labels of the test data
            
        Returns:
            acc (float64) : accuracy
            spe (float64) : specivity
            sen (float64) : sensitivy
            
        """
        
        nop, noi = self.getDistribution(y)
        
        
        tp = 0 # correctly classified preictal
        ti = 0 # correctly classified interictal
        p = nop # number of preictal
        i = noi # number of interictal
        fp = 0  # mistakenly classified preictal
        fi = 0 # mistakenly classified interictal
        
        
        assert p + i == x.shape[0] # preictal samples plus interictal must equal the total.
        sampleNo = p + i
        
        for i in range(sampleNo):
            # (1,0) label stands for preictal
            # (0,1) label stands for interictal (minority)
            
            sampleData = x[i, :, :] if x.ndim == 3 else x[i,:]
            sampleLabel = y[i, :] if y.ndim == 2 else y[i]
            
            if isinstance(self.layers[0], Conv2D):
                isInterictal = True if np.argmax(sampleLabel) == np.argmax([0,1]) else False
            else:
                isInterictal = True if sampleLabel == -1 else False
            
            guess = self.predict(sampleData, sampleLabel)
            
            if isInterictal:
                if np.argmax(guess) == np.argmax((0, 1)):
                    ti += 1 #correctly classified
                else:
                    fp += 1 #falsely classified
            else:
                if np.argmax(guess) == np.argmax((1, 0)):
                    tp += 1
                else:
                    fi += 1
                    
        acc = (tp + ti) / sampleNo
        spe = ti / (fp + ti)
        sen = tp / (tp + fi)
        
        if verbose:
            print(f"Accuracy = {acc}")
            print(f"Specivity = {spe}")
            print(f"Sensitivity = {sen}")
        
        return acc, spe, sen
        
        
    def getDistribution(self, y):
        
        """
        Given a sample with 52 preictal periods and 48 interictal periods, 
        this function returns 52, 48, respectively. 
        
        Param:
            y (float64) : labels
            
        Returns:
            nop (int16) : no of preictal periods
            noi (int16) : no of interictal periods
            
        """
        
        nop = noi = 0
        sampleNo = y.shape[0]
        
        # (1,0) label stands for preictal
        # (0,1) label stands for interictal (minority)

        
        for i in range(sampleNo):
            if isinstance(self.layers[0], Conv2D):
                if np.argmax(y[i, :]) == np.argmax((1,0)):
                    nop += 1
                else:
                    noi += 1
            else:
                if y[i] == 1:
                    nop += 1
                else:
                    noi += 1
                    
        return nop, noi
        
        
        
        
        
        
        
        
        