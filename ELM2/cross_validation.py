import numpy as np
import pandas as pd

class CrossValidation:
    """
    """

    def __init__(self, k_splits):
        self._k_splits = k_splits
    
    def get_k(self):
        return self._k_splits
    
    def split(self, table):
        """
        Parameters
        ----------
        table: a dataframe

        Returns
        -------
        two calculated integer indexes begin and end of the window

        """
        # implementation from the sickit learn library
        pass
    
    def KFold(self, table):
        pass