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
    print(nota)
    return nota,y2

def convertedatas(data):
    data.reverse()
    data = np.asarray(data)
    print(data.size)
    for i in range(0,data.size,1):
      data[i] = i
    data2 = data.reshape(-1,1)
    data2 = data2.astype(np.float)
    print(data2)
    #scaler =  StandardScaler()
    #data3 = scaler.fit_transform(data3)
    return data2

def preprocessing(valor):
    valor.reverse()
    valor = np.asarray(valor)
    valor = valor.reshape(-1,1)
    imputer = SimpleImputer(missing_values=-1,strategy='mean')
    valor = imputer.fit_transform(valor)
    #scaler =  StandardScaler()
    #valor = scaler.fit_transform(valor)
    return valor

def polynomial_regression(x,y,a):
    polyLinearRegression = LinearRegression()
    polyLinearRegression.fit(x,y)
    return polyLinearRegression

def showPlot(XPoints, yPoints, x,y):
    print(x)
    plt.scatter(x,y,color = 'red') #Mostra os pontos reais dos dados
    plt.plot(XPoints,yPoints,color = 'blue') #Mostra os pontos preditos pelo modelo
    plt.title("Comparando pontos reais com a reta produzida pela regressão polinomial")
    plt.xlabel("Data")
    plt.ylabel("Confirmados")
    plt.show()

if __name__ == "__main__":
    arq = """ [
        {
            "data": "11/07/2020",
            "confirmados": "334",
            "monitorados": "202",
            "curados": "125",
            "óbitos": "6"
        },
        {
            "data": "10/07/2020",
            "confirmados": "318",
            "monitorados": "187",
            "curados": "125",
            "óbitos": "6"
        },
        {
            "data": "09/07/2020",
            "confirmados": "296",
            "monitorados": "174",
            "curados": "117",
            "óbitos": "5"
        },
        {
            "data": "08/07/2020",
            "confirmados": "281",
            "monitorados": "161",
            "curados": "115",
            "óbitos": "5"
        },
        {
            "data": "07/07/2020",
            "confirmados": "254",
            "monitorados": "139",
            "curados": "110",
            "óbitos": "5"
        },
        {
            "data": "06/07/2020",
            "confirmados": "238",
            "monitorados": "128",
            "curados": "105",
            "óbitos": "5"
        },
        {
            "data": "05/07/2020",
            "confirmados": "230",
            "monitorados": "121",
            "curados": "105",
            "óbitos": "4"
        },
        {
            "data": "04/07/2020",
            "confirmados": "219",
            "monitorados": "120",
            "curados": "95",
            "óbitos": "4"
        },
        {
            "data": "03/07/2020",
            "confirmados": "217",
            "monitorados": "118",
            "curados": "95",
            "óbitos": "4"
        },
        {
            "data": "02/07/2020",
            "confirmados": "202",
            "monitorados": "108",
            "curados": "90",
            "óbitos": "4"
        },
        {
            "data": "01/07/2020",
            "confirmados": "175",
            "monitorados": "86",
            "curados": "85",
            "óbitos": "4"
        },
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
        },
        {
            "data": "31/05/2020",
            "confirmados": "60",
            "monitorados": "15",
            "curados": "43",
            "óbitos": "2"
        },
        {
            "data": "30/05/2020",
            "confirmados": "59",
            "monitorados": "14",
            "curados": "43",
            "óbitos": "2"
        },
        {
            "data": "29/05/2020",
            "confirmados": "59",
            "monitorados": "16",
            "curados": "41",
            "óbitos": "2"
        },
        {
            "data": "28/05/2020",
            "confirmados": "59",
            "monitorados": "16",
            "curados": "41",
            "óbitos": "2"
        },
        {
            "data": "27/05/2020",
            "confirmados": "55",
            "monitorados": "15",
            "curados": "38",
            "óbitos": "2"
        },
        {
            "data": "26/05/2020",
            "confirmados": "55",
            "monitorados": "15",
            "curados": "38",
            "óbitos": "2"
        },
        {
            "data": "25/05/2020",
            "confirmados": "53",
            "monitorados": "13",
            "curados": "38",
            "óbitos": "2"
        },
        {
            "data": "24/05/2020",
            "confirmados": "53",
            "monitorados": "21",
            "curados": "30",
            "óbitos": "2"
        },
        {
            "data": "23/05/2020",
            "confirmados": "53",
            "monitorados": "21",
            "curados": "30",
            "óbitos": "2"
        },
        {
            "data": "22/05/2020",
            "confirmados": "50",
            "monitorados": "18",
            "curados": "30",
            "óbitos": "2"
        },
        {
            "data": "21/05/2020",
            "confirmados": "49",
            "monitorados": "18",
            "curados": "29",
            "óbitos": "2"
        },
        {
            "data": "20/05/2020",
            "confirmados": "47",
            "monitorados": "16",
            "curados": "29",
            "óbitos": "2"
        },
        {
            "data": "19/05/2020",
            "confirmados": "46",
            "monitorados": "25",
            "curados": "19",
            "óbitos": "2"
        },
        {
            "data": "18/05/2020",
            "confirmados": "40",
            "monitorados": "21",
            "curados": "17",
            "óbitos": "2"
        },
        {
            "data": "17/05/2020",
            "confirmados": "38",
            "monitorados": "23",
            "curados": "13",
            "óbitos": "2"
        },
        {
            "data": "16/05/2020",
            "confirmados": "38",
            "monitorados": "23",
            "curados": "13",
            "óbitos": "2"
        },
        {
            "data": "15/05/2020",
            "confirmados": "38",
            "monitorados": "25",
            "curados": "13",
            "óbitos": "2"
        },
        {
            "data": "14/05/2020",
            "confirmados": "35",
            "monitorados": "21",
            "curados": "13",
            "óbitos": "1"
        },
        {
            "data": "13/05/2020",
            "confirmados": "35",
            "monitorados": "78",
            "curados": "13",
            "óbitos": "1"
        },
        {
            "data": "12/05/2020",
            "confirmados": "26",
            "monitorados": "63",
            "curados": "13",
            "óbitos": "1"
        },
        {
            "data": "11/05/2020",
            "confirmados": "26",
            "monitorados": "62",
            "curados": "09",
            "óbitos": "1"
        },
        {
            "data": "10/05/2020",
            "confirmados": "19",
            "monitorados": "57",
            "curados": "09",
            "óbitos": "1"
        },
        {
            "data": "09/05/2020",
            "confirmados": "19",
            "monitorados": "58",
            "curados": "09",
            "óbitos": "1"
        },
        {
            "data": "08/05/2020",
            "confirmados": "18",
            "monitorados": "61",
            "curados": "09",
            "óbitos": "1"
        },
        {
            "data": "07/05/2020",
            "confirmados": "18",
            "monitorados": "59",
            "curados": "09",
            "óbitos": "1"
        },
        {
            "data": "06/05/2020",
            "confirmados": "18",
            "monitorados": "50",
            "curados": "09",
            "óbitos": "1"
        },
        {
            "data": "05/05/2020",
            "confirmados": "18",
            "monitorados": "40",
            "curados": "09",
            "óbitos": "1"
        },
        {
            "data": "04/05/2020",
            "confirmados": "16",
            "monitorados": "44",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "03/05/2020",
            "confirmados": "15",
            "monitorados": "41",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "02/05/2020",
            "confirmados": "15",
            "monitorados": "39",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "01/05/2020",
            "confirmados": "15",
            "monitorados": "37",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "30/04/2020",
            "confirmados": "15",
            "monitorados": "36",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "29/04/2020",
            "confirmados": "13",
            "monitorados": "25",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "28/04/2020",
            "confirmados": "12",
            "monitorados": "23",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "27/04/2020",
            "confirmados": "09",
            "monitorados": "19",
            "curados": "4",
            "óbitos": "1"
        },
        {
            "data": "26/04/2020",
            "confirmados": "08",
            "monitorados": "21",
            "curados": "3",
            "óbitos": "1"
        },
        {
            "data": "25/04/2020",
            "confirmados": "08",
            "monitorados": "22",
            "curados": "3",
            "óbitos": "1"
        },
        {
            "data": "24/04/2020",
            "confirmados": "-1",
            "monitorados": "-1",
            "curados": "-1",
            "óbitos": "-1"
        },
        {
            "data": "23/04/2020",
            "confirmados": "-1",
            "monitorados": "-1",
            "curados": "-1",
            "óbitos": "-1"
        },
        {
            "data": "22/04/2020",
            "confirmados": "07",
            "monitorados": "23",
            "curados": "3",
            "óbitos": "1"
        },
        {
            "data": "21/04/2020",
            "confirmados": "06",
            "monitorados": "34",
            "curados": "3",
            "óbitos": "1"
        },
        {
            "data": "20/04/2020",
            "confirmados": "06",
            "monitorados": "36",
            "curados": "3",
            "óbitos": "1"
        },
        {
            "data": "19/04/2020",
            "confirmados": "06",
            "monitorados": "37",
            "curados": "3",
            "óbitos": "1"
        },
        {
            "data": "18/04/2020",
            "confirmados": "-1",
            "monitorados": "-1",
            "curados": "-1",
            "óbitos": "-1"
        },
        {
            "data": "17/04/2020",
            "confirmados": "6",
            "monitorados": "31",
            "curados": "3",
            "óbitos": "1"
        },
        {
            "data": "16/04/2020",
            "confirmados": "6",
            "monitorados": "31",
            "curados": "2",
            "óbitos": "1"
        },
        {
            "data": "15/04/2020",
            "confirmados": "4",
            "monitorados": "24",
            "curados": "0",
            "óbitos": "1"
        },
        {
            "data": "14/04/2020",
            "confirmados": "4",
            "monitorados": "20",
            "curados": "0",
            "óbitos": "1"
        },
        {
            "data": "13/04/2020",
            "confirmados": "4",
            "monitorados": "18",
            "curados": "0",
            "óbitos": "1"
        },
        {
            "data": "12/04/2020",
            "confirmados": "4",
            "monitorados": "32",
            "curados": "0",
            "óbitos": "1"
        },
        {
            "data": "11/04/2020",
            "confirmados": "4",
            "monitorados": "30",
            "curados": "0",
            "óbitos": "1"
        },
        {
            "data": "10/04/2020",
            "confirmados": "-1",
            "monitorados": "-1",
            "curados": "-1",
            "óbitos": "-1"
        },
        {
            "data": "09/04/2020",
            "confirmados": "1",
            "monitorados": "29",
            "curados": "0",
            "óbitos": "1"
        },
        {
            "data": "08/04/2020",
            "confirmados": "1",
            "monitorados": "30",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "07/04/2020",
            "confirmados": "1",
            "monitorados": "-1",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "06/04/2020",
            "confirmados": "0",
            "monitorados": "16",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "05/04/2020",
            "confirmados": "0",
            "monitorados": "15",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "04/04/2020",
            "confirmados": "0",
            "monitorados": "-1",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "03/04/2020",
            "confirmados": "0",
            "monitorados": "11",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "02/04/2020",
            "confirmados": "0",
            "monitorados": "-1",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "01/04/2020",
            "confirmados": "0",
            "monitorados": "23",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "31/03/2020",
            "confirmados": "0",
            "monitorados": "19",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "30/03/2020",
            "confirmados": "0",
            "monitorados": "16",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "29/03/2020",
            "confirmados": "0",
            "monitorados": "15",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "28/03/2020",
            "confirmados": "0",
            "monitorados": "15",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "27/03/2020",
            "confirmados": "0",
            "monitorados": "15",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "26/03/2020",
            "confirmados": "0",
            "monitorados": "15",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "25/03/2020",
            "confirmados": "0",
            "monitorados": "11",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "24/03/2020",
            "confirmados": "0",
            "monitorados": "11",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "23/03/2020",
            "confirmados": "0",
            "monitorados": "7",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "22/03/2020",
            "confirmados": "0",
            "monitorados": "7",
            "curados": "0",
            "óbitos": "0"
        },
        {
            "data": "21/03/2020",
            "confirmados": "0",
            "monitorados": "3",
            "curados": "0",
            "óbitos": "0"
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
    xtreinodata, xtestedata, ytreinoconf, ytesteconf = train_test_split(datas, confirmados, test_size = 0.1)
    xtreinodata, xtestedata, ytreinomoni, ytestemoni = train_test_split(datas, monitorados, test_size = 0.1)
    xtreinodata, xtestedata, ytreinocura, ytestecura = train_test_split(datas, curados, test_size = 0.1)
    xtreinodata, xtestedata, ytreinomortes, ytestemortes= train_test_split(datas, mortes, test_size = 0.1)
    xtreinoaux = xtreinodata
    xtesteaux =  xtestedata
    xtestedata = xtestedata
    xtreinodata = xtreinodata
    pr1 = polynomial_regression(xtreinodata,ytreinoconf,1)
    pr2 = polynomial_regression(xtreinodata,ytreinomoni,1)
    pr3 = polynomial_regression(xtreinodata, ytreinocura,1)
    pr4 = polynomial_regression(xtreinodata,ytreinomortes,1)
#    print(xtreinodata,ytreinoconf)
    nt1,y1 = evaluate(pr1,xtestedata,ytesteconf)
    nt2,y2 = evaluate(pr2,xtestedata,ytestemoni)
    nt3,y3 = evaluate(pr3,xtestedata,ytestecura)
    nt4,y4= evaluate(pr4,xtestedata,ytestemortes)
    #print(ytesteconf.shape,xtestedata.shape,y1.shape)
    showPlot(xtestedata,ytesteconf,xtestedata,y1)
    showPlot(xtestedata,ytestemoni,xtestedata,y2)
    showPlot(xtestedata, ytestecura,xtestedata,y3)
    showPlot(xtestedata,ytestemortes,xtestedata,y4)
