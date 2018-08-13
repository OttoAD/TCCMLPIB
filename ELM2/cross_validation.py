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
        fold_sizes = (table.shape[0] // self._k_splits) * np.ones(self._k_splits, dtype=np.int) #creates an array of sizes
        fold_sizes[:table.shape[0] % self._k_splits] += 1 #corretcs the size if number of samples is odd 
        current = 0

        for fold_size in fold_sizes: #iterates over the array of size k_fold
            begin, end = current, current + fold_size #slides the window 
            yield begin, end #yeilds(returns) the current fold
            current = end #updates the beginning of the fold

    def KFold(self, table, target):
        """"
        Parameters
        ----------
        table: a dataframe

        Returns
        -------
        
        """
        for start,end in self.split(table): #iterates over the split dataset
            training = pd.concat([table.iloc[:start,:],table.iloc[end:,:]]) #creats the training dataset
            target_train = pd.concat([target.iloc[:start,:],target.iloc[end:,:]])
            test = table.iloc[start:end,:] #creates the test dataset
            target_result = target.iloc[start:end]
            yield training, test, target_train, target_result#return both datasets

            #fazer o treino/teste 70/30
            #calcular a média do erro kfold pra cada iteração e retornar