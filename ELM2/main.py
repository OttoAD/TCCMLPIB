import data as dt
import regression as lr
import numpy as np
import pandas as pd

# Main method definition
def main():

    # Importing data
    inputData = dt.Data()

    #Normalizing input data skipping the first row(bias row)
    indexTable = inputData.read_data(fileName = "./data/Dados.csv", index = "Periodo", dates = True, period = "Q")
    originalGDP = inputData.read_data(fileName = "./data/PIB.csv", index = "Periodo", dates = True, period = "Q")
    
    # Shuffling the rows of both dataframes
    shuffledIndex = inputData.shuffle(indexTable)
    shuffledGDP = inputData.shuffle(originalGDP)
    #print(shuffledIndex)
    #print(shuffledGDP)
    
    ########## MULTIVARIATE LINEAR REGRESSION ##########
    #lm = lr.LinearModel()

    #predictedGDP = shuffledIndex.dot(lm.linear_regression(table = shuffledIndex, target = shuffledGDP))
    #print(predictedGDP)

    ########## NEURAL NETWORK: EXTREME LEARNING MACHINE ##########
    #Weights are calculated by a random normal distribution with the following values:
    #Mean value : 0
    #Standard deviation: 1
    #Samples: shuffledGDP.shape[0] which is the value of all the rows/observations in the input dataframe
    # The number of neurons is arbitrary and the data value is the input value
    neuralNetwork = nt.NeuralNetwork(neurons = 700,
                                    data = shuffledIndex)
    # PANDAS: Not only must the shapes of DF1 and DF2 be correct, but also the COLUMN names of DF1 must match the INDEX names of DF2.
    # The NumPy dot function does no such thing. It will just compute the matrix product based on the values in the underlying arrays.
    # The weights matrix was multiplied by the transposed matrix of the input index values.

    #TRAIN,PREDICT AND ANALYZE

    ########## K FOLD CROSS VALIDATION  ##########


if __name__ == "__main__":
    main()