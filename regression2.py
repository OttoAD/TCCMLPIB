import numpy as np
import scipy as sp
import pandas as pd

# ----- LEITURA DO ARQUIVO -----
print ("----- IMPORTAÇÃO -----")
file = "Data.xls"
data = pd.ExcelFile(file)
#print(data.sheet_names)
df = data.parse(0)
pib = data.parse(1)

print ("----- NORMALIZAÇÃO -----")
desvio = df.std()
normalizada = df.copy()
normalizada.iloc[:,2:] = df.iloc[:,2:].div(desvio[2:])

print ("----- REGRESSÃO MULTIVARIADA -----")
norm_matrix = normalizada.iloc[:,2:].as_matrix()
print("----- MATRIZ -----\n"+str(norm_matrix)+"\n")

x_trans = norm_matrix.transpose() #X^t
print("----- TRANSPOSTA -----\n"+str(x_trans)+"\n")
x_mult = x_trans.dot(norm_matrix) #X^t * X
print("----- MULTIPLICADA -----\n"+str(x_mult)+"\n")
x_inv = np.linalg.inv(x_mult) #x_mult ^ -1
print("----- INVERSA -----\n"+str(x_inv)+"\n")
w = (x_inv.dot(x_trans)).dot(pib)
print("----- W* -----\n"+str(w)+"\n")
y_aproximado = norm_matrix.dot(w)
print("----- Y APROXIMADO -----\n"+str(y_aproximado)+"\n")