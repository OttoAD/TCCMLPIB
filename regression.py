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
#print(normalizada)
print ("----- REGRESSÃO MULTIVARIADA -----")
normalizada.set_index(["Ano","Periodo (Trimestre)"],inplace=True)
print("\n----- NORMALIZADA -----\n"+str(normalizada))
normalizada_trans = normalizada.T #X^t
print("\n----- TRANSPOSTA -----\n"+str(normalizada_trans))
normalizada_mult = normalizada_trans.dot(normalizada)#X^t * X
print("\n----- MULTIPLICAÇÃO -----\n"+str(normalizada_mult))
normalizada_inv = pd.DataFrame(np.linalg.inv(normalizada_mult),normalizada_mult.columns,normalizada_mult.index)#x_mult ^ -1
print("\n----- INVERSA -----\n"+str(normalizada_inv))
pib.set_index(["Ano","Periodo (Trimestre)"],inplace=True)
print("\n----- PIB -----\n"+str(pib))
w = (normalizada_inv.dot(normalizada_trans)).dot(pib)
print("\n----- MULTIPLICAÇÃO 2: W* -----\n"+str(w))
y_aproximado = normalizada.dot(w)
print("----- Y APROXIMADO -----\n"+str(y_aproximado)+"\n")
