import numpy as np
import matplotlib.pyplot as plt

def estimar_coeficientes(x, y):
    #parte problematica
    b_1 = 0 #(N*som(xj*yj)-(som(xj))*som(yj))/(N*som(xj²)-som(xj)²)
    b_0 = 0 #(som(yj) - b_1*som(xj))/N
    #---------------------------
 
    return(b_0, b_1)

def plotar_grafico(x,y,b):
    plt.scatter(x, y, color = "r", marker = "x")
 
    y_previsto = b[0] + b[1]*x
 
    plt.plot(x, y_previsto, color = "b")

    plt.show()

def main():
    y = np.array([460,232,315,150,402])
    x = np.array([2014,1516,1534,1800,1760])
    #y = np.round(np.random.normal(5,2,100),2)
    #x = np.arange(0,100,1)
    b = estimar_coeficientes(x, y)

    plotar_grafico(x,y,b)

if __name__ == "__main__":
    main()