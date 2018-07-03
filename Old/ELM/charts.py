import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PLOTTING METHODS
# Line plot
def plot(originalData, predictedData):
    data = pd.concat([originalData, predictedData], axis=1)
    data.columns = ['Real','Estimado']
    chart = data.plot(title = 'PIB Real x Estimado',xticks=data.index,rot=45)
    chart.set_xlabel('Periodo (Trimestral)')
    chart.set_ylabel('PIB (%)')
    chart.grid(True, which='minor', axis='x' )
    chart.grid(True, which='major', axis='y' )
    plt.show()

#Bar plot
def barPlot(originalData, predictedData):
    data = pd.concat([originalData, predictedData], axis=1)
    data.columns = ['Real','Estimado']
    chart = data.plot(title = 'PIB Real x Estimado', kind='bar')
    chart.set_ylabel('PIB (%)')
    plt.grid(alpha=0.3)
    chart.set_axisbelow(True)
    for values in chart.patches:
        chart.annotate(str(np.round(values.get_height(),2)), (values.get_x() * 1.01, values.get_height() * 1.015), ha='center', va='center', fontsize = 6)

    plt.show()