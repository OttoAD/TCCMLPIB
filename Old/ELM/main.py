import data as dt
import regression as lr
import network as nt
import numpy as np
import charts
import pandas as pd

# Main method definition
def main():
    # Instantiating classes
    inputData = dt.Data()
    regressionModel = lr.LinearRegression()

    # Importing data
    indexTable = inputData.readData(fileName = "./data/Dados.csv", index = "Periodo", dates = True, period = "Q")
    originalGDP = inputData.readData(fileName = "./data/PIB.csv", index = "Periodo", dates = True, period = "Q")

    #Normalizing input data skipping the first row(bias row)
    indexTable = inputData.normalize(indexTable,skip = 1)

    # Calculating the last row of a parcentage of data to be trained
    rowIndex = regressionModel.calculateIndex(indexTable, percentage = 0.75) 

    # Shuffling the rows of both dataframes
    permutation = np.random.permutation(indexTable.index)
    shuffledIndex = indexTable.reindex(permutation)
    shuffledGDP = originalGDP.reindex(permutation)

    ########## MULTIVARIATE LINEAR REGRESSION ##########
    # Generating weights table for the linear regression
    weights = regressionModel.trainModel(rowIndex = rowIndex,
                         data = shuffledIndex,
                         target = shuffledGDP)

    # Applying the GDP prediction: Training model is [:rowIndex,:] and validation model is [rowIndex:,:]
    predictedGDP = shuffledIndex.iloc[:rowIndex,:].dot(weights)

    #Calculating the absolute average error between the original value and the predicted value. Must pay attention to the indexes sent
    GDPError = regressionModel.avgerror(predictedGDP.iloc[:rowIndex,:],shuffledGDP.iloc[:rowIndex,:])

    #DEBUG PRINTS
    #print("\n----- MATRIZ DE PESOS -----\n" + str(weights))
    #print("\n----- PIB PREVISTO -----\n" + str(predictedGDP))
    #print("\nERRO:\n" + str(GDPError))

    ########## NEURAL NETWORK: EXTREME LEARNING MACHINE ##########
    #Weights are calculated by a random normal distribution with the following values:
    #Mean value : 0
    #Standard deviation: 1
    #Samples: shuffledGDP.shape[0] which is the value of all the rows/observations in the input dataframe
    # The number of neurons is arbitrary and the data value is the input value
    neuralNetwork = nt.NeuralNetwork(neurons = 400,
                                    data = shuffledIndex)

    # PANDAS: Not only must the shapes of DF1 and DF2 be correct, but also the COLUMN names of DF1 must match the INDEX names of DF2.
    # The NumPy dot function does no such thing. It will just compute the matrix product based on the values in the underlying arrays.
    # The weights matrix was multiplied by the transposed matrix of the input index values.
    
    #print(neuralNetwork.weights())
    #H = neuralNetwork.weights().dot(shuflledIndex.T) #20x10 * transposta(89x10) = 20x89
    #print("\n----- H = W*X -----\n" + str(H))
    
    #H = H.apply(neuralNetwork.sigmoid)
    #print("\n----- SIGMOIDE(H) -----\n" + str(H))

    #COPIANDO O CODIGO DA REGRESSÃO LINEAR
    # Generating weights table for the linear regression
    #Hinverse = regressionModel.pseudoInverse(H)
    #print(H.shape)
    #print(Hinverse.shape)
    #print(shuffledGDP.shape)
    #Beta = Hinverse.dot(shuffledGDP)
    H = neuralNetwork.feedforward(shuffledIndex)
    print(H)
    Hinverse = regressionModel.pseudoInverse(H)
    #print(H.shape)
    #print(Hinverse.shape)
    #print(shuffledGDP.shape)
    
    Beta = Hinverse.T.dot(shuffledGDP) #está diferente da formula original
    print(Beta)

    PESOS = neuralNetwork.weights()
    H2 = PESOS.dot(shuffledIndex.T)
    H2 = H2.apply(neuralNetwork.sigmoid)
    print(H2)
    ESTIMADO = Beta.T.dot(H2)
    #print(ESTIMADO)

    GDPError2 = regressionModel.avgerror(ESTIMADO.T,shuffledGDP)
    combinado = pd.concat([shuffledGDP,ESTIMADO.T],axis=1)
    print(combinado)
    print("\nERRO ABAIXO\n")
    print(GDPError2)
    #print(teste.shape)

    # neuralNetwork2 = nt.NeuralNetwork(neurons = 20,
    #                         data = shuflledIndex,
    #                         weights = neuralNetwork.weights())

    # H2 = neuralNetwork2.weights().dot(shuflledIndex.T) #20x10 * transposta(89x10) = 20x89
    
    # print(H2)
    
    #print("\n----- H = W*X -----\n" + str(H))
    
#    H2 = H2.apply(neuralNetwork2.sigmoid)

    # print(Beta.T.shape)
    # print(H2.shape)
    # resultado = Beta.T.dot(H2)
    # print(resultado)

if __name__ == "__main__":
    main()