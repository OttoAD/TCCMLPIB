import numpy as np
import pandas as pd

class Data:
    """
    Classe que modela a importação e manipulação dos dados.
    
    Methods
    ----------
    get_permutation :
        Getter method which returns the _permutation attribute
    set_permutation :
        Setter method which assigns a value to the _permutation attribute
    normalize :
        A method which divides all the values of a dataframe by its standard deviation
    shuffle :
        A method which reindex all the observations in a dataframe
    mean_average_error :
        A method that calculates the average absolute error between two dataframes
    """

    def __init__(self):
        self._permutation = []  # The permutation is set after the first shuffle

    def get_permutation(self):
        """
        This method is a getter which returns the _permutation attribute.

        Parameters
        ----------
        None
        
        Returns
        ---------- 
        The permutation utilized for shuffling the observations in a dataframe
        """
        return self._permutation
    
    def set_permutation(self, permutation):
        """
        This method is a setter which assigns a value to the _permutation attribute.

        Parameters
        ----------
        permutation 
        
        Returns
        ----------
        None
        """
        self._permutation = permutation
    
    def read_data(self, fileName, index = None, separator = ",", dates = False, period = None):
        """
        A method which takes as argument a file to be read and its descriptors
        
        Parameters
        ---------- 
        fileName :
            Or file path, it is the name or path of the file to be read
        index :
            This parameter indicates which column should be read as the dataframe index. It is set as None by default
        separator :
            Indicates which separator is being used in the CSV file. Is is set as a comma by default
        dates :
            Parameter that indicates if dates should be parsed or not. It is set as False by default
        period :
            This parameter descibes the period of the input (daily, monthly, yearly, etc). It is set as None by default

        Returns
        ----------
        The file read and stored as a dataframe structre
        """

        if period == None:
            return pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
        else:
            data = pd.read_csv(fileName, index_col = index, sep = separator, parse_dates = dates)
            data.index = data.index.to_period(period)
            return data

    def normalize(self, table, begin = 0, end = 0):
        """
        A method which normalizes a given dataframe, in other words, it divides all the values of the dataframe by its standard deviation.       

        Parameters
        ----------
        table :
            An input dataframe to be normalized
        begin :
            An index indicating the beginning of the normalization window. It is 0 by default
        end :
            An index indicating the end of the normalization window. It is 0 by default

        Returns
        ----------
        table :
            The normalized input dataframe
        """
        
        if begin == 0 and end == 0:
            table = table.div(table.std())
        elif begin != 0 and end == 0:
            table.iloc[:,begin:] = table.iloc[:,begin:].div(table.iloc[:,begin:].std())
        elif end > begin:
            table.iloc[:,begin:end] = table.iloc[:,begin:end].div(table.iloc[:,begin:end].std())
        
        return table
    
    def shuffle(self, table):
        """
        A method reponsible for reindexing all the observations in a given dataframe according to a permutation

        Parameters
        ----------
        table :
            The input dataframe to be shuffled
        
        Returns
        ----------
        table:
            A shuffled dataframe where all its observations have been reindexed by a permutation
        """

        if self._permutation == []:
            self._permutation = np.random.permutation(table.index)
        
        return table.reindex(self._permutation)
    
    def mean_average_error(self, table1, table2):
        """
        Calculates the average absolute error between two dataframes.

        Parameters
        ----------
        table1 and table2

        Returns
        ----------
        A number representing the average absolute error for the given dataframes
        """
        return table1.subtract(table2).abs().mean()
    
    ##### METODOS AUXILIARES A SEREM REPENSADOS #####
    def compare_results(self, result_table, original_table):
        """
        Concatenate two dataframes around the Y axis for value comparison.
        
        Parameters
        ----------
        results_table
        original_table
        
        Returns
        ----------
        A new dataframe of the concatenated input tables
        """
        return pd.concat([original_table, result_table], axis = 1)
    
    def analyze(self, result_table, original_table):
        index = original_table.shape[0] - result_table.shape[0]
        print(" ----- ORIGINAL X ESTIMADO ----- \n"+ str(self.compare_results(result_table, original_table.iloc[index:,:])) + "\n")
        print(" ----- ERRO ----- \n" + str(self.mean_average_error(result_table,original_table.iloc[index:,:])) + "\n")
    