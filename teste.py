import numpy as np
import scipy as sp
import matplotlib as mp
import matplotlib.pyplot as plt
#import pandas
#import sklearn

print ("----- INICIO DO PROGRAMA -----")
#estado inicial do array com 100 elementos
#y = np.round(np.random.normal(5,1,100),2)
#x = np.arange(0,100,1)
y = [460,232,315]
x_original = [[1,2104,5],
            [1,1416,3],
            [1,1534,3]]

x_matrix = np.array(x_original)
x_trans = x_matrix.transpose() #X^t
x_mult = x_trans*x_matrix #X^t * X
x_inv = np.linalg.inv(x_mult) #x_mult ^ -1
w = x_inv*x_trans*y
y_aproximado = w.transpose()*x_original
print(w)
#plt.scatter(x_matrix,y,marker = "x")
#plt.show()

