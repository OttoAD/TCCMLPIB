import numpy as np
import pandas as pd
import data as dt
import regression as lr
import network as nt
import cross_validation as cv

# Main method definition
def main():

    # Importing data
    inputData = dt.Data()

    #Normalizing input data skipping the first row(bias row)
    indexTable = inputData.read_data(fileName = "./data/Dados.csv", index = "Periodo", dates = True, period = "Q")
    originalGDP = inputData.read_data(fileName = "./data/PIB.csv", index = "Periodo", dates = True, period = "Q")
    
    #Normalizing input data skipping the first row(bias row)
    indexTable = inputData.normalize(indexTable,begin = 1)
    
    # Shuffling the rows of both dataframes
    shuffledIndex = inputData.shuffle(indexTable)
    shuffledGDP = inputData.shuffle(originalGDP)
    #print(shuffledIndex)
    #print(shuffledGDP)
    
    ########## MULTIVARIATE LINEAR REGRESSION ##########
    lm = lr.LinearModel()

    predictedGDP = lm.linear_regression(table = shuffledIndex, target = shuffledGDP)
    #print(predictedGDP)

    ########## NEURAL NETWORK: EXTREME LEARNING MACHINE ##########
    #Weights are calculated by a random normal distribution with the following values:
    #Mean value : 0
    #Standard deviation: 1
    #Samples: shuffledGDP.shape[0] which is the value of all the rows/observations in the input dataframe
    # The number of neurons is arbitrary and the data value is the input value
    neuralNetwork = nt.NeuralNetwork(neurons = 700, C = 10,
                                    data = shuffledIndex)
    # PANDAS: Not only must the shapes of DF1 and DF2 be correct, but also the COLUMN names of DF1 must match the INDEX names of DF2.
    # The NumPy dot function does no such thing. It will just compute the matrix product based on the values in the underlying arrays.
    # The weights matrix was multiplied by the transposed matrix of the input index values.

    #TRAIN,PREDICT AND ANALYZE
    neuralNetwork.train(table = shuffledIndex, target = shuffledGDP, train_percentage = 70)
    predictedGDP = neuralNetwork.test(table = shuffledIndex, test_percentage = 30)
    #print(predictedGDP)
    #print(shuffledGDP)
    #inputData.analyze(predictedGDP.T,shuffledGDP)

    ########## K FOLD CROSS VALIDATION  ##########
    neuralNetwork.validate(table = shuffledIndex, target = shuffledGDP, k_samples = 5)
    
    #REMOVER AS LINHAS ABAIXO
    #kval = cv.CrossValidation(5)
    #opa = kval.KFold(table = shuffledIndex)
    # training, test = next(opa)
    # print(training)
    # print(test)
    # neuralNetwork.train(table = training, target = shuffledGDP.iloc[18:], train_percentage = 100)
    # result = neuralNetwork.test(test, test_percentage = 100)
    # print(result)
    # error = inputData.mean_average_error(result.T, shuffledGDP)
    # print(error)

    # training, test = next(opa)
    # print(training)
    # print(test)
    # a = pd.concat([shuffledGDP.iloc[:18],shuffledGDP.iloc[36:]])
    # neuralNetwork.train(table = training, target = a, train_percentage = 100)

    # training, test = next(opa)
    # print(training)
    # print(test)
    # a = pd.concat([shuffledGDP.iloc[:18],shuffledGDP.iloc[36:]])
    # neuralNetwork.train(table = training, target = a, train_percentage = 100)

    #neuralNetwork.validate(shuffledIndex, shuffledGDP)
if __name__ == "__main__":
    main()