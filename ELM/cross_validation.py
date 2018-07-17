import numpy as np
import scipy as sp
import pandas as pd

class CrossValidation:
    """
    """

    def __init__(self, k):
        self._k = k
        self._performance = None
        self._step = None

        pass
    
    def get_k(self):
        return self._k

    def get_performance(self):
        return self._performance
    
    def get_step(self):
        return self._step

    def calculate_step(self, length):
        """
        """
        
        self._step = np.round(length/self._k).astype(int)
    
    def kfold(self, table):
        """
        """

        if self._step == None:
            self.calculate_step(table.shape[0])

        

        pass
    
    def cross_validate(self, table):
        """
        """
        pass
    


    # implementar o shuffle nesta classe?