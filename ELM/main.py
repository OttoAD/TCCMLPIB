import data as dt
import regression as lr
import network as nt
import numpy as np
import elm_plot as pl
import pandas as pd
import cross_validation as cv

# Main method definition
def main():
    # Instantiating classes
    inputData = dt.Data()
    lm = lr.LinearModel()

    # Importing data
    indexTable = inputData.read_data(fileName = "./data/Dados.csv", index = "Periodo", dates = True, period = "Q")
    originalGDP = inputData.read_data(fileName = "./data/PIB.csv", index = "Periodo", dates = True, period = "Q")

    #Normalizing input data skipping the first row(bias row)
    indexTable = inputData.normalize(indexTable,begin = 1)

    # Shuffling the rows of both dataframes
    shuffledIndex = inputData.shuffle(indexTable)
    shuffledGDP = inputData.shuffle(originalGDP)

    ########## MULTIVARIATE LINEAR REGRESSION ##########




    ########## NEURAL NETWORK: EXTREME LEARNING MACHINE ##########
    #Weights are calculated by a random normal distribution with the following values:
    #Mean value : 0
    #Standard deviation: 1
    #Samples: shuffledGDP.shape[0] which is the value of all the rows/observations in the input dataframe
    # The number of neurons is arbitrary and the data value is the input value

    neuralNetwork = nt.NeuralNetwork(neurons = 750,
                                    data = shuffledIndex)
    
    # PANDAS: Not only must the shapes of DF1 and DF2 be correct, but also the COLUMN names of DF1 must match the INDEX names of DF2.
    # The NumPy dot function does no such thing. It will just compute the matrix product based on the values in the underlying arrays.
    # The weights matrix was multiplied by the transposed matrix of the input index values.

    neuralNetwork.train(shuffledIndex, shuffledGDP)
    predictedGDP = neuralNetwork.test(shuffledIndex)
    inputData.analyze(predictedGDP.T,shuffledGDP)

    ########## K FOLD CROSS VALIDATION  ##########
    #kval = cv.CrossValidation(5)
    #opa = kval.KFold(shuffledIndex)

    neuralNetwork.validate(shuffledIndex, shuffledGDP)

    #training,test = next(opa)
    #print(training)
    #print(test)
if __name__ == "__main__":
    main()