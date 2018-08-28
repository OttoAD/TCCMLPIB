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

    #SHUFFLING
#    lm.train(table = shuffledIndex, target = shuffledGDP)
#    predictedRegressionGDP = lm.test(table = shuffledIndex)
#    print(predictedRegressionGDP)
#    lm.validate(table = shuffledIndex, target = shuffledGDP) #KFOLD

    #NO SHUFFLING
    lm.train(table = indexTable, target = originalGDP)
    predictedRegressionGDP = lm.test(table = indexTable)
#    print(predictedRegressionGDP)
#    lm.validate(table = indexTable, target = originalGDP) #KFOLD

    ########## NEURAL NETWORK: EXTREME LEARNING MACHINE ##########
    #Weights are calculated by a random normal distribution with the following values:
    #Mean value : 0
    #Standard deviation: 1
    #Samples: shuffledGDP.shape[0] which is the value of all the rows/observations in the input dataframe
    # The number of neurons is arbitrary and the data value is the input value
    
    #SHUFFLE
 #   neuralNetwork = nt.NeuralNetwork(neurons = 800, C = 10,
 #                                   data = shuffledIndex)

    #NO SHUFFLE                                
#    neuralNetwork = nt.NeuralNetwork(neurons = 800, C = 10, data = indexTable)
    # PANDAS: Not only must the shapes of DF1 and DF2 be correct, but also the COLUMN names of DF1 must match the INDEX names of DF2.
    # The NumPy dot function does no such thing. It will just compute the matrix product based on the values in the underlying arrays.
    # The weights matrix was multiplied by the transposed matrix of the input index values.

    #TRAIN,PREDICT AND ANALYZE

    # SHUFFLE
 #   neuralNetwork.train(table = shuffledIndex, target = shuffledGDP, train_percentage = 70)
 #   predictedGDP = neuralNetwork.test(table = shuffledIndex, test_percentage = 30)
 #   print(predictedGDP)
 #   print(shuffledGDP)
    #inputData.analyze(predictedGDP.T,shuffledGDP)
 #   neuralNetwork.validate(table = shuffledIndex, target = shuffledGDP, k_samples = 5) #KFOLD

    #NO SHUFFLING
#    neuralNetwork.train(table = indexTable, target = originalGDP)
#    predictedELMGDP = neuralNetwork.test(table = indexTable)
#    print(predictedELMGDP.T)
    #neuralNetwork.validate(table = indexTable, target = originalGDP, k_samples = 5) #KFOLD
    ########## PLOT  ##########

    #PLOT PARA COMPARAÇÃO DA REGRESSÃO COM A ELM
    #OBS: não usar shuffle
    #OBS2: os valores DOS PESOS da ELM são chutados a cada vez q são criados
#    predictedRegressionGDP = predictedRegressionGDP.rename(columns = {"PIB real (%)":"Regression"})
#    predictedELMGDP = predictedELMGDP.transpose().rename(columns = {"PIB real (%)":"ELM"})
#    data = pd.concat([predictedRegressionGDP,predictedELMGDP], axis = 1)
#    print(data)

    for i in range(1,11):
        neuralNetwork = nt.NeuralNetwork(neurons = 800, C = 10, data = indexTable)
        neuralNetwork.train(table = indexTable, target = originalGDP)
        predictedELMGDP = neuralNetwork.test(table = indexTable)
        predictedRegressionGDP = predictedRegressionGDP.rename(columns = {"PIB real (%)":"Regressão"})
        predictedELMGDP = predictedELMGDP.transpose().rename(columns = {"PIB real (%)":"ELM"})
        data = pd.concat([predictedRegressionGDP,predictedELMGDP,originalGDP.iloc[63:]], axis = 1)
        data.to_csv('./fig/data' + str(i)+'.csv', sep=',', encoding='utf8')
        inputData.plot(data,'./fig/line' + str(i) + '.png')
        inputData.barPlot(data,'./fig/bar' + str(i) + '.png')

if __name__ == "__main__":
    main()