import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
df = pd.read_csv("https://firebasestorage.googleapis.com/v0/b/angular-html-11.appspot.com/o/datasetFinalDesBalanceado.csv?alt=media&token=ff6a6468-2294-44ba-bcb1-e71efaacc54b",delimiter=";")

def dataset_balanceado(df):
    indigenas = df[:48]
    blancos = df[48:97]
    negros = df[97:142]
    mestizos = df[142:]
    indigenas = indigenas.sample(n = 45,random_state=5)
    blancos = blancos.sample(n = 45, random_state=5)
    negros = negros.sample(n = 45,random_state=5)
    mestizos = mestizos.sample(n = 45, random_state=5)
    
    columnas = [i for i in range(0,137)]
    dataset_final = np.concatenate([indigenas,blancos,negros,mestizos])
    
    df = pd.DataFrame(data = dataset_final,columns=columnas)
    
    X = np.array(df.iloc[:,0:136])
    y = np.array(df.iloc[:,[136]])
    
    return X,y  

def dataset_desbalanceado(df):
    X = np.array(df.iloc[:,0:135])
    y = np.array(df.iloc[:,[136]])
    
    return X,y


def pantalla_P():
    st.write("""
# Dataset of Ethnic Facial Images of Ecuadorian People
Este conjunto de datos está compuesto por imágenes faciales étnicas de personas ecuatorianas.

Cada imagen tiene metadatos de sus respectivas medidas biométricas: cejas, ojos, nariz y boca. Además, este conjunto de datos contiene información relacionada con: datos personales, características étnicas y ascendencia.

 Los metadatos sobre la información facial étnica y biométrica se obtuvieron mediante dos técnicas:

1. Aplicación de algoritmos de inteligencia artificial: utilizados para la detección, identificación y biometría facial para la extracción de puntos de interés antropométricos. Estas medidas se almacenan en un archivo XML, con los puntos que detectó el algoritmo.

2. Encuestas: se utilizan para obtener información personal, características étnicas, ascendencia, autodenominación étnica, etc.

Se tomaron las encuestas y el conjunto de datos de imágenes, en el período del 1 de abril al 30 de mayo de 2019. Se consideró la situación demográfica de las principales etnias del Ecuador y se tomó la muestra en las ciudades de Quito, Guayaquil, Ibarra y Latacunga. El conjunto de datos compromete 430 imágenes de personas (autodenominadas) de los siguientes grupos étnicos:

• Afroecuatoriano (50)

• Descendiente europeo (50)

• Indígena (50)

• Mestizo (280)

Las imágenes tienen una resolución de 1920x1280 píxeles a todo color y el escenario que se definió para la captura tiene las siguientes características:

• Fondo: blanco

• Distancia del individuo hacia la cámara: 1,10 [m]

• Brillo: iluminación frontal y lateral

• Postura del sujeto: recta con expresión facial neutra y libre de objetos que obstaculicen el rostro.

Todas las personas que participaron en el proyecto dieron su consentimiento para el uso de su fotografía e información personal con fines académicos, que incluyen: revistas científicas, presentaciones y repositorios académicos digitales.

Acerca de los archivos:

• Las imágenes se distribuyen en cuatro carpetas, según el grupo étnico: afroecuatorianos, descendientes de europeos, indígenas y mestizos.

• Las especificaciones técnicas de la cámara, utilizadas en la construcción de este conjunto de datos, se describen en el archivo Cam_spec.csv (carpeta "Metadata").

• Cada uno de los archivos XML generados por origen étnico se incluyen en la carpeta "Metadatos".

• Las respuestas de la encuesta aplicada y las características de las variables se incluyen en los archivos: Ethnic_facial_characteristics_answers.csv y Variables_characteristics.csv, respectivamente, dentro de la carpeta "Metadata".
""")
    
    pagina_oficial = "https://ndownloader.figshare.com/articles/8266730/versions/3"

    st.write(f" <a href = {pagina_oficial}> Descargar Dataset Completo </a>",unsafe_allow_html=True)
def pantalla_balanceada():
    st.write('''
             # **Algoritmos de Aprendizaje Supervisado con Clases Balanceadas**
             Las Clases Balanceadas se Dividen en
             
              • Indígena (45)
              
              • Blancos o Descendiente europeo (45)
              
              • Afroecuatorianos o Negros (45)
              
              • Mestizo (45)
             
              **Con un total de 180 instancias**
             ''')
def pantalla_desbalanceada():
    st.write('''
             # **Algoritmos de Aprendizaje Supervisado con Clases Desbalanceadas**
             Las Clases Desbalanceadas se Dividen en
             
              • Indígena (48)
              
              • Blancos o Descendiente europeo (49)
              
              • Afroecuatorianos o Negros (45)
              
              • Mestizo (273)
             
              **Con un total de 415 instancias**
             ''')   
def sidebar():
    balance = st.sidebar.selectbox("Desea Usar Clases",("Descripcion del Dataset","Balanceadas","Desbalanceadas"))
    
    return balance
def sidebarClasificado():
        clasificador = st.sidebar.selectbox("Seleccione el algoritmo", ("KNN", "RANDOM FOREST", "MAQUINA DE SOPORTE VECTORIAL", "NAIVE BAYES","REDES NEURONALES","REGRESION LOGISTICA"))
        return clasificador

def knn(X,y,vecinos):
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, np.ravel(y_train,order='C'))
        scores.append(knn.score(X_test, y_test))
    plt.figure(figsize=(12,7))
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    st.pyplot()
    
    n_neighbors = vecinos
 
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train,np.ravel(y_train,order='C'))
    st.write('Accuracy de K-NN classifier en el test es : {:.2f}'
         .format(knn.score(X_test, y_test)))
    y_pred = knn.predict(X_test)
    st.write("Matriz de Confusion \n",confusion_matrix(y_test, y_pred))
    st.write("Precision de cada clase ",precision_score(y_test, y_pred,average=None))
    st.write("Recall de cada clase ", recall_score(y_test, y_pred,average=None))
    st.write("F1-score de cada clase ",f1_score(y_test, y_pred,average=None))
    
def random_forest(X,y):
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)
    model = RandomForestClassifier(n_estimators=100,max_depth=4)
    model.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = model.predict(X_test)
    st.write('Accuracy Usando Random Forest es : {:.2f}'
     .format(model.score(X_test, y_test)))
    st.write("Matriz de Confusion \n",confusion_matrix(y_test, y_pred))
    st.write("Precision de cada clase ",precision_score(y_test, y_pred,average=None))
    st.write("Recall de cada clase ", recall_score(y_test, y_pred,average=None))
    st.write("F1-score de cada clase ",f1_score(y_test, y_pred,average=None))

def naive_bayes(X,y):
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)
    gnb = GaussianNB()
    gnb.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = gnb.predict(X_test)
    st.write('Accuracy Usando Naive Bayes es : {:.2f}'.format(gnb.score(X_test, y_test)))
    st.write("Matriz de Confusion \n",confusion_matrix(y_test, y_pred))
    st.write("Precision de cada clase ",precision_score(y_test, y_pred,average=None))
    st.write("Recall de cada clase ", recall_score(y_test, y_pred,average=None))
    st.write("F1-score de cada clase ",f1_score(y_test, y_pred,average=None))
    
    
def regresion_logistica(X,y):
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)
    model = linear_model.LogisticRegression(max_iter=100,solver='liblinear')
    model.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = model.predict(X_test)
    st.write('Accuracy Usando Regresion Logistica es : {:.2f}'.format(model.score(X_test, y_test)))
    st.write("Matriz de Confusion \n",confusion_matrix(y_test, y_pred))
    st.write("Precision de cada clase ",precision_score(y_test, y_pred,average=None))
    st.write("Recall de cada clase ", recall_score(y_test, y_pred,average=None))
    st.write("F1-score de cada clase ",f1_score(y_test, y_pred,average=None))
    

def red_neuronal(X,y):
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)
    mlp = MLPClassifier(hidden_layer_sizes=(100,10),random_state=1, max_iter=500, alpha=0.0001,
                        solver = 'adam', tol = 0.000000001) ## 2 capas ocultas la una con 100 unidades y la otra con 10
    mlp.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = mlp.predict(X_test)
    st.write('Accuracy Usando Red Neuronal es : {:.2f}'.format(mlp.score(X_test, y_test)))
    st.write("Matriz de Confusion \n",confusion_matrix(y_test, y_pred))
    st.write("Precision de cada clase ",precision_score(y_test, y_pred,average=None))
    st.write("Recall de cada clase ", recall_score(y_test, y_pred,average=None))
    st.write("F1-score de cada clase ",f1_score(y_test, y_pred,average=None))

def maquina_Soporte(X,y):
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)
    svr = svm.SVC(C = 1000,kernel='linear',decision_function_shape='ovo')
    svr.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = svr.predict(X_test)
    
    st.write('Accuracy Usando Red Neuronal es : {:.2f}'.format(svr.score(X_test, y_test)))
    st.write("Matriz de Confusion \n",confusion_matrix(y_test, y_pred))
    st.write("Precision de cada clase ",precision_score(y_test, y_pred,average=None))
    st.write("Recall de cada clase ", recall_score(y_test, y_pred,average=None))
    st.write("F1-score de cada clase ",f1_score(y_test, y_pred,average=None))
    
        
balanceada = sidebar()



if balanceada == "Descripcion del Dataset":
    pantalla_P()
elif balanceada == "Balanceadas":
    pantalla_balanceada()
    clasificador = sidebarClasificado()
    X,y = dataset_balanceado(df)
    st.write("""
             ***
             """) 
    st.write(f"<h2> Seleccionaste el algoritmo : {clasificador} </h1> ", unsafe_allow_html = True)
    if clasificador == "KNN":
        st.write("<h2> Grafico donde muestran los puntos con mayor precision </h2> ",unsafe_allow_html=True)
        knn(X,y,12)
    elif clasificador == "RANDOM FOREST":
        random_forest(X,y)
    elif clasificador == "MAQUINA DE SOPORTE VECTORIAL":
        maquina_Soporte(X,y)
    elif clasificador == "NAIVE BAYES":
        naive_bayes(X, y)
    elif clasificador == "REGRESION LOGISTICA":
        regresion_logistica(X, y)
    elif clasificador == "REDES NEURONALES":
        red_neuronal(X,y)
    
        
    
    
elif balanceada == "Desbalanceadas":
    pantalla_desbalanceada()
    clasificador = sidebarClasificado()
    X,y = dataset_desbalanceado(df)
    st.write("""
             ***
             """)
    st.write(f"<h2> Seleccionaste el algoritmo : {clasificador} </h1> ", unsafe_allow_html = True)
    if clasificador == "KNN":
        st.write("<h2> Grafico donde muestran los puntos con mayor precision </h2> ",unsafe_allow_html=True)
        knn(X,y,15)
    elif clasificador == "RANDOM FOREST":
        random_forest(X,y)
    elif clasificador == "MAQUINA DE SOPORTE VECTORIAL":
        maquina_Soporte(X,y)
    elif clasificador == "NAIVE BAYES":
        naive_bayes(X, y)
    elif clasificador == "REGRESION LOGISTICA":
        regresion_logistica(X, y)
    elif clasificador == "REDES NEURONALES":
        red_neuronal(X,y)
 
