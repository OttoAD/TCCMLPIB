import numpy as np
import scipy as sp
import matplotlib as mp
import matplotlib.pyplot as plt
#import pandas
#import sklearn

print ("----- MULTIVARIADO -----")
#estado inicial do array com 100 elementos
#y = np.round(np.random.normal(5,1,100),2)
#x = np.arange(0,100,1)

#Matriz copiada do coursera
y = [460,232,315]
x = [[1,2104,5],
    [1,1416,3],
    [1,1534,3]]

    #problema com matriz não quadradas?

x_matrix = np.array(x)
print("MATRIZ:\n"+str(x_matrix)+"\n")
x_trans = x_matrix.transpose() #X^t
print("TRANSPOSTA:\n"+str(x_trans)+"\n")
x_mult = x_trans.dot(x_matrix) #X^t * X
print("MULTIPLICADA:\n"+str(x_mult)+"\n")
x_inv = np.linalg.inv(x_mult) #x_mult ^ -1
print("INVERSA:\n"+str(x_inv)+"\n")
w = (x_inv.dot(x_trans)).dot(y)
print("W*:\n"+str(w)+"\n")
y_aproximado = (w.transpose()).dot(x)
np.round(y_aproximado,2) #nao funciona
print("Y APROXIMADO:\n"+str(y_aproximado)+"\n")
#como plotar N dimensões? Precisa? Como sei que funcionou?
#plt.scatter(x_matrix,y,marker = "x")
#plt.plot()
#plt.show()
