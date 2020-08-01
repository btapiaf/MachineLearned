import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

valor = []
datos2 =pd.read_csv("Compi_sin_NA.csv",header=None)
datos = datos2.drop([0,1,2],axis = 1)

#Indígenas: 1
#Blancos: 2
#Negros: 3
#Mestizos: 4

for i in range(6,273,4):
    valor.append(i)
    
valores_fin = datos.drop([i for i in valor],axis = 1)

valores = valores_fin.replace('[array]','',regex=True).astype(str)
s = []
for i in range(5,273,4):
    s.append(i)
    
s.append(273)
final = valores.drop([i for i in s],axis = 1)
final = final.replace("[( )]",'',regex=True).astype(str)
final = final.replace("\[",'',regex=True).astype(str)
final = final.replace("\]",'',regex=True).astype(int)


y = datos2.iloc[:,[0]].astype(int)
X = np.array(final)

dataset = np.append(X, y, axis=1)

indigenas = dataset[:48]
blancos = dataset[48:97]
negros = dataset[97:142]
mestizos = dataset[142:]




mestizos,aleatorio_mestizos = train_test_split(mestizos,test_size = 0.17,random_state=42)


dataset_final = np.concatenate([indigenas,blancos,negros,aleatorio_mestizos])
columnas = [i for i in range(0,137)]

print(len(aleatorio_mestizos)," ",len(indigenas)," ", len(blancos)," ",len(negros))
df = pd.DataFrame(data = dataset_final,columns=columnas)
df.to_csv("datasetFinal.csv",sep=',', encoding='utf-8', index=False)


