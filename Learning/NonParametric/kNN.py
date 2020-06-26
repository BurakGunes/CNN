# -*- coding: utf-8 -*-
"""
This class implements the non-parametric pattern recognition algorithm 
k-nearest neighbors.
"""

from data_util import PCA
import numpy as np

class kNN:
    
    
    
    def __init__(self):
        
        """Initialize the variables of the kNN algorithm."""
        
        self.k = -1 #number of neighbors to be used for classification
        
    def compile_model(self, input_size, output_size):
        
        """Model parameters must be set before any prediction can take place.
        
        Params:
            input_size () : input size as a tuple
            output_size (int16) : output size, e.g. for binary, this is 2.
        
        """
        
        self.input_size = input_size
        self.output_size = output_size
        
        return True
    
    def set_k(self, k):
        
        """The k parameter of the algorithm is set using this function.
        
        Param:
            k (int16) : number of neighbors to be used during the learning.
            
        """
        #for example, given a 2-class prediction, 2 neighbors would be ambigous
        assert k % self.output_size != 0 
        
        self.k = k
        
        return True
    
    
    def set_distance_function(self, fun):
        
        """Among the implemented distance functions, the user can choose
        the one desired.
        
        Param:
            fun (str) : distance function. Currently supported is Euclidian.
        
        """
        
        self.distance_function = fun
        
        return True

    
    def fit_model(self, X_train, Y_train):
        
        """Inputs are saved. 
        
        Params:
            X_train (float64) : input vector
            Y_train (float64) : corresponding labels of the input vector
        """
       
        self.xt = X_train
        self.yt = Y_train
        self.N = self.xt.shape[0] #note the data count N.
        
        return True
    
    
    def predict(self, x, y, verbose = False, ppp = None):
        
        """Given the single input vectors x and y, this function predicts the 
        class of the input"""
        
        distances = np.zeros((self.N))
        
        for i in range(self.N):
            distances[i] = self.distance(self.xt[i, :], x, ppp)
            
        idx = np.argsort(distances)
        idx_k = idx[:self.k]
        
        labels = self.yt[idx_k]
        
        pre = inter = 0
        
        for label in labels:
            if label == 1:
                pre += 1
            else:
                inter += 1
        
        
        if pre > inter:
            return 1
        else:
            return -1
        
        return None
    
    def test(self, X_test, Y_test, verbose = False, pp = None):
        
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
        
        nop, noi = self.getDistribution(Y_test)
        
        
        tp = 0 # correctly classified preictal
        ti = 0 # correctly classified interictal
        p = nop # number of preictal
        i = noi # number of interictal
        fp = 0  # mistakenly classified preictal
        fi = 0 # mistakenly classified interictal
        
        
        assert p + i == Y_test.shape[0] # preictal samples plus interictal must equal the total.
        sampleNo = p + i
        
        for i in range(sampleNo):
            # (1,0) label stands for preictal
            # (0,1) label stands for interictal (minority)
            
            sampleData = X_test[i,:]
            sampleLabel = Y_test[i]
            
            isInterictal = True if sampleLabel == -1 else False
            
            guess = self.predict(sampleData, sampleLabel, ppp = pp)
            
            if isInterictal:
                if guess == -1:
                    ti += 1 #correctly classified
                else:
                    fp += 1 #falsely classified
            else:
                if guess == 1:
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
        
        # (1,0) or  1 label stands for preictal
        # (0,1) or -1 label stands for interictal (minority)

        
        for i in range(sampleNo):
            if y[i] == 1:
                nop += 1
            else:
                noi += 1
                
        return nop, noi
        
    
    def distance(self, x1, x2, p = None):
        
        if self.distance_function == "L1":
            return self.manhattan_distance(x1, x2)
        elif self.distance_function == "L2":
            return self.euclidean_distance(x1, x2)
        elif self.distance_function == "M":
            return self.minkowski_distance(x1, x2, p)

        
        return None
   
    
    def euclidean_distance(self, x1, x2):
        
        """Calculates the euclidean distance between two samples."""
        
        q = x1 - x2 #calculate the norm between them
        euc = np.sqrt(np.dot(q.T, q))
        
        return euc
    
    def manhattan_distance(self, x1, x2):
        
        """Calculates the manhattan distance between two samples."""
        
        q = np.abs(x1 - x2)
        l1 = np.sum(q)
        
        return l1
    
    def minkowski_distance(self, x1, x2, p):
        
        """Calculates the minkowski distance between two samples."""
        
        q = np.abs(x1 - x2)
        qp = np.power(q, p)
        qps = np.sum(qp)
        
        M = np.power(qps, 1/p)
        return M
        
