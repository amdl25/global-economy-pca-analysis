import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd


def corelograma(matrice=None, dec=1, titlu='Corelograma', valMin=-1, valMax=1):
    plt.figure(titlu, figsize=(30, 25))
    plt.title(titlu, fontsize=14, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrice, dec), cmap='bwr', vmin=valMin, vmax=valMax, annot=True)

def componentePrincipale(valoriProprii=None, titlu='Varianta explicata de componentele principale',
                         etichetaX='Componente principale', etichetaY='Valori proprii'):
    plt.figure(titlu, figsize=(11, 8))
    plt.title(titlu, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(xlabel=etichetaX, fontsize=14, color='b', verticalalignment='top')
    plt.ylabel(ylabel=etichetaY, fontsize=14, color='b', verticalalignment='bottom')
    componente = ['C'+str(i+1) for i in range(len(valoriProprii))]
    plt.axhline(y=1, color='r')
    plt.plot(componente, valoriProprii, 'bo-')

def afisare():
    plt.show()