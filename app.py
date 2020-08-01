# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:50:16 2020

@author: USUARIO
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

st.write("""
         # APRENDIZAJE SUPERVISADO
         ## ALGORITMOS DE CLASIFICACION Y REGRESION
""")

nombre_data_set = st.write("## Dataset of Ethnic facial images of Ecuadorian people")

st.write('''
         
         ## la Variable Dependiente se Divide de la Siguiente Forma:
         ### Indígenas: 1 
         ### Blancos: 2 
         ### Negros: 3 
         ### Mestizos: 4 
    
         ''')

clasificador = st.sidebar.selectbox("Seleccione el algoritmo", ("KNN", "RANDOM FOREST", "SVM", "NAIVE BAYES","REDES NEURONALES","REGRESION LOGISTICA"))

st.write("ha seleccionado el siguiente algoritmo: ",clasificador)

def get_dataset(nombre_data_set):
    df = pd.read_csv('https://firebasestorage.googleapis.com/v0/b/angular-html-11.appspot.com/o/datasetFinal.csv?alt=media&token=3558c2ec-e92f-472f-9cb8-9f282c25b8e2')
    
    X = np.array(df.iloc[:,0:136])
    y = np.array(df.iloc[:,[136]])
    return X,y


    
X, y = get_dataset(nombre_data_set)


st.write("shape", X.shape)


def parametros(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("Número de vecinos", 1, 15)
        params["K"] = K 

    elif clf_name == "SVM":
        C = st.sidebar.slider("Parámetro de regularización C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "NAIVE BAYES":
        G = ("G")
    elif clf_name =="REDES NEURONALES":
        R = ("R")
    elif clf_name =="REGRESION LOGISTICA":
        variables = st.sidebar.selectbox('Seleccione las variables a predecir',("INDIGENAS Y BLANCOS",""))
        params['variables'] = variables
    elif clf_name =="RANDOM FOREST":
        max_depth = st.sidebar.slider("profundidad máxima del árbol", 2, 15)
        n_estimators = st.sidebar.slider("El número de árboles", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators 
    return params
   
params = parametros(clasificador)    

def get_clasificadores(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "NAIVE BAYES":
        clf = GaussianNB(priors=None, var_smoothing=1e-9)
    elif clf_name =="REDES NEURONALES":
        clf = MLPClassifier(activation='identity',alpha=1e-5,hidden_layer_sizes=(5, 2), verbose=False, random_state=1)
    elif clf_name == "REGRESION LOGISTICA":
        clf = LogisticRegression(max_iter = 1000,solver="newton-cg")
    elif clf_name =="RANDOM FOREST":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=1234)
    return clf
    
clf = get_clasificadores(clasificador, params)

#clasificacion

#Indígenas: 1 0:47
#Blancos: 2 48:96
#Negros: 3 97:141
#Mestizos: 4 142:
    
    

# x_blancos = X[48:96]
# y_blancos = y[48:96]
# x_negros = X[97:141]
# y_negros = y[97:141]
# x_mestizos = X[142:]
# y_mestizos = y[142:]    

if clasificador =='REGRESION LOGISTICA':
    X = X[0:96] #indigenas y blancos
    y = y[0:96]
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf.fit(X_train, np.ravel(y_train,order='C'))
    y_predict = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    pre = precision_score(y_test, y_predict,average='macro')
    rep = classification_report(y_test, y_predict)
    st.write(f"Regresion = {clasificador}")
    st.write(f"accuracy = {acc}")
    st.write(f"precision = {pre}")
    st.write(f"reporte clasificador   = {rep}")
    st.write("La matriz de Confusion es la Siguiente: ",confusion_matrix(y_test, y_predict))
    
    
else: 
          
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, np.ravel(y_train,order='C'))
    y_predict = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    pre = precision_score(y_test, y_predict,average='macro')
    rep = classification_report(y_test, y_predict)
    st.write(f"Clasificador = {clasificador}")
    st.write(f"accuracy = {acc}")
    st.write(f"precision = {pre}")
    st.write(f"reporte clasificador   = {rep}")
    st.write("La matriz de Confusion es la Siguiente: ",confusion_matrix(y_test, y_predict))




    
