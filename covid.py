#-*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
#vai comer sua chata linda
def evaluate(pr,x,y1):
    y2 = pr.predict(x)
    nota = r2_score(y1,y2)
    return nota,y2

def convertedatas(data):
    data.reverse()
    data = np.asarray(data)
    labelencoder_X = LabelEncoder()
    data2 = labelencoder_X.fit_transform(data)    
    data3 = data2.reshape(-1,1)
    return data3

def preprocessing(valor):
    valor.reverse()
    valor = np.asarray(valor)
    valor = valor.reshape(-1,1)
    imputer = SimpleImputer(missing_values=-1,strategy='mean')
    valor = imputer.fit_transform(valor)
    scaler =  StandardScaler()
    valor = scaler.fit_transform(valor)
    return valor

def transform(x,a):
    polynomialFeatures = PolynomialFeatures(degree = a)
    XPolynomial = polynomialFeatures.fit_transform(x)
    return XPolynomial

def polynomial_regression(x,y,a):
    polyLinearRegression = LinearRegression().fit(x, y)
    return polyLinearRegression

def showPlot(XPoints, yPoints, XLine, yLine):
    plt.scatter(XPoints,yPoints,color = 'red') #Mostra os pontos reais dos dados
    plt.plot(XLine,yLine,color = 'blue') #Mostra os pontos preditos pelo modelo
    plt.title("Comparando pontos reais com a reta produzida pela regressão polinomial")
    plt.xlabel("Data")
    plt.ylabel("Confirmados")
    plt.show()

if __name__ == "__main__":
    arq = """
    [
        {
            "data": "30/06/2020",
            "confirmados": "159",
            "monitorados": "75",
            "curados": "80",
            "óbitos": "4"
        },
        {
            "data": "29/06/2020",
            "confirmados": "156",
            "monitorados": "72",
            "curados": "80",
            "óbitos": "4"
        },
        {
            "data": "28/06/2020",
            "confirmados": "143",
            "monitorados": "65",
            "curados": "75",
            "óbitos": "3"
        },
        {
            "data": "27/06/2020",
            "confirmados": "141",
            "monitorados": "64",
            "curados": "74",
            "óbitos": "3"
        },
        {
            "data": "26/06/2020",
            "confirmados": "133",
            "monitorados": "56",
            "curados": "74",
            "óbitos": "3"
        },
        {
            "data": "25/06/2020",
            "confirmados": "122",
            "monitorados": "45",
            "curados": "74",
            "óbitos": "3"
        },
        {
            "data": "24/06/2020",
            "confirmados": "116",
            "monitorados": "39",
            "curados": "74",
            "óbitos": "3"
        },
        {
            "data": "23/06/2020",
            "confirmados": "107",
            "monitorados": "34",
            "curados": "70",
            "óbitos": "3"
        },
        {
            "data": "22/06/2020",
            "confirmados": "106",
            "monitorados": "33",
            "curados": "70",
            "óbitos": "3"
        },
        {
            "data": "21/06/2020",
            "confirmados": "103",
            "monitorados": "31",
            "curados": "68",
            "óbitos": "3"
        },
        {
            "data": "20/06/2020",
            "confirmados": "96",
            "monitorados": "26",
            "curados": "66",
            "óbitos": "3"
        },
        {
            "data": "19/06/2020",
            "confirmados": "95",
            "monitorados": "25",
            "curados": "66",
            "óbitos": "3"
        },
        {
            "data": "18/06/2020",
            "confirmados": "89",
            "monitorados": "19",
            "curados": "66",
            "óbitos": "3"
        },
        {
            "data": "17/06/2020",
            "confirmados": "80",
            "monitorados": "10",
            "curados": "66",
            "óbitos": "3"
        },
        {
            "data": "16/06/2020",
            "confirmados": "80",
            "monitorados": "16",
            "curados": "60",
            "óbitos": "3"
        },
        {
            "data": "15/06/2020",
            "confirmados": "80",
            "monitorados": "16",
            "curados": "60",
            "óbitos": "3"
        },
        {
            "data": "14/06/2020",
            "confirmados": "77",
            "monitorados": "13",
            "curados": "60",
            "óbitos": "3"
        },
        {
            "data": "13/06/2020",
            "confirmados": "77",
            "monitorados": "15",
            "curados": "58",
            "óbitos": "3"
        },
        {
            "data": "12/06/2020",
            "confirmados": "75",
            "monitorados": "16",
            "curados": "55",
            "óbitos": "3"
        },
        {
            "data": "11/06/2020",
            "confirmados": "71",
            "monitorados": "12",
            "curados": "55",
            "óbitos": "3"
        },
        {
            "data": "10/06/2020",
            "confirmados": "70",
            "monitorados": "11",
            "curados": "55",
            "óbitos": "2"
        },
        {
            "data": "09/06/2020",
            "confirmados": "67",
            "monitorados": "10",
            "curados": "55",
            "óbitos": "2"
        },
        {
            "data": "08/06/2020",
            "confirmados": "67",
            "monitorados": "11",
            "curados": "54",
            "óbitos": "2"
        },
        {
            "data": "07/06/2020",
            "confirmados": "66",
            "monitorados": "10",
            "curados": "54",
            "óbitos": "2"
        },
        {
            "data": "06/06/2020",
            "confirmados": "66",
            "monitorados": "11",
            "curados": "52",
            "óbitos": "2"
        },
        {
            "data": "05/06/2020",
            "confirmados": "65",
            "monitorados": "10",
            "curados": "52",
            "óbitos": "2"
        },
        {
            "data": "04/06/2020",
            "confirmados": "64",
            "monitorados": "12",
            "curados": "50",
            "óbitos": "2"
        },
        {
            "data": "03/06/2020",
            "confirmados": "63",
            "monitorados": "13",
            "curados": "48",
            "óbitos": "2"
        },
        {
            "data": "02/06/2020",
            "confirmados": "60",
            "monitorados": "10",
            "curados": "48",
            "óbitos": "2"
        },
        {
            "data": "01/06/2020",
            "confirmados": "60",
            "monitorados": "12",
            "curados": "46",
            "óbitos": "2"
        }
    ]
    """
    data = []
    confirmados = []
    monitorados = []
    curados = []
    mortes = []
    lista = json.loads(arq)
    for i,nome in enumerate(lista):
      dicionario = lista[i]
      data.append(dicionario["data"])
      confirmados.append(dicionario["confirmados"])
      monitorados.append(dicionario["monitorados"])
      curados.append(dicionario["curados"])
      mortes.append(dicionario["óbitos"])
    confirmados = preprocessing(confirmados)
    monitorados = preprocessing(monitorados)
    curados = preprocessing(curados)
    mortes = preprocessing(mortes)
    datas = convertedatas(data)
    xtreinodata, xtestedata, ytreinoconf, ytesteconf = train_test_split(datas, confirmados, test_size = 0.2)
    xtreinodata, xtestedata, ytreinomoni, ytestemoni = train_test_split(datas, monitorados, test_size = 0.2)
    xtreinodata, xtestedata, ytreinocura, ytestecura = train_test_split(datas, curados, test_size = 0.2)
    xtreinodata, xtestedata, ytreinomortes, ytestemortes= train_test_split(datas, mortes, test_size = 0.2)
    print(xtestedata)   
    xtestedata = transform(xtestedata,2)
    xtestedata = transform(xtreinodata,2)
    print(xtestedata)
    pr1 = polynomial_regression(xtreinodata,ytreinoconf,2)
    pr2 = polynomial_regression(xtreinodata,ytreinomoni,2)
    pr3 = polynomial_regression(xtreinodata, ytreinocura,2)
    pr4 = polynomial_regression(xtreinodata,ytreinomortes,2)
#    print(xtreinodata,ytreinoconf)
    nt1,y1 = evaluate(pr1,xtestedata,ytesteconf)
    nt2,y2 = evaluate(pr2,xtestedata,ytestemoni)
    nt3,y3 = evaluate(pr3,xtestedata, ytestecura)
    nt4,y4= evaluate(pr4,xtestedata,ytestemortes)
   # print(xtestedata,ytesteconf.shape,xtestedata.shape,y1.shape)
    showPlot(xtestedata,ytesteconf,xtestedata,y1)
    showPlot(xtestedata,ytestemoni,xtestedata,y2)
    showPlot(xtestedata, ytestecura,xtestedata,y3)
    showPlot(xtestedata,ytestemortes,xtestedata,y4)
