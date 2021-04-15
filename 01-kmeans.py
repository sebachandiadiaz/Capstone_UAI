#%% md

# Actividad k-means (color quantization)

#Una de los usos de k-means es color quantization. Para ello se necesita realizar los siguientes pasos
#1. Cargar una imagen a su gusto (ojalá que no sea una imagen muy grande)
#2. Transformar el set de datos a una matriz de nx3
#3. Aplicar k-means (con k de 1 a 100 y y busque el mejor valor de k), use n_init=1
#4. Reemplazar cada pixel por el color correspondiente
#5. Volver a cambiar el tamaño de la imagen
#6. Generar la imagen

#Realice un proceso de color quantization a una imagen de su gusto.

#%%

#Instalando las librerias requeridas
#import sys
#!{sys.executable} -m pip install numpy, pandas, plotnine, sklearn

#%%

import numpy as np
import pandas as pd
from plotnine import *
from sklearn.cluster import KMeans

#%%

#1 Cargar una imagen a su gusto
#El siguiente código permite cargar una imagen a su gusto (ojalá jpg)
from PIL import Image
im = np.array(Image.open('C:/Users/sebas/OneDrive/Escritorio/Magister Data Science UAI/9. Machine Learning/Ejecicios_Jupyter/venv/UAI.jpg')) #descomente esta línea
#PATH es el directorio donde se encuentra la biblioteca
#nombre es el nombre de la imagen
#extension es la extensión de la imagen
print(im.shape) #Verifique el tamaño de la imagen

#%%

#2 transformar el set de datos
#Para transformar el set de datos genere una copia de la imagen con nuevoArreglo=nombreArreglo.copy()
im2 = im.copy()
#Posteriormente use la función nuevaForma=np.reshape(imagenOriginal,(dim1,dim2,...,dimn))
newForm = np.reshape(im2,(im2.shape[0]*im2.shape[1], 3))
#Atención la multiplicación del tamaño de las imagenes original y final tiene que ser las mismas

#%%

#3 Aplicar k-means (con k de 1 a 100 y y busque el mejor valor de k), use n_init=1

sse = []
for k in range(1, 100):
    km = KMeans(n_clusters=k, n_init=1, max_iter=100)
    km.fit(newForm)
    sse.append(km.inertia_)

#%%

#3.1 Código de ejemplo para la búsqueda de k
(ggplot()+aes(x=range(1, 100),y=sse)+theme_bw()+geom_line()+labs(x="Número de clusters",y="WCD")
 +scale_x_continuous()+coord_cartesian(xlim=[0,60],ylim=[0,100000000]))

#%%

#4 Reemplazar cada pixel por el color correspondiente
#Aplique k-means con el k seleccionado
km2 = KMeans(n_clusters=15, n_init=1, max_iter=100)
km2.fit(newForm)

#itere sobre el número de clusteres seleccionados
    #Cambie los pixeles correspondiente al cluster i, por el centroide del cluster i

newForm2 = newForm

for i in range(len(newForm)):
    newForm2[i] = km2.cluster_centers_[km2.labels_[i]]

print(newForm2)

#%%

#5 Volver a cambiar el tamaño de la imagen, al tamaño original
newForm2 = np.reshape(newForm2,(im2.shape[0],im2.shape[1], 3))

#%%

#6) Grabar la nueva imagen
import matplotlib.pyplot as mpl
mpl.imsave('C:/Users/sebas/OneDrive/Escritorio/Magister Data Science UAI/9. Machine Learning/Ejecicios_Jupyter/venv/UAI_2.jpg', newForm2) #Comente esta línea
#PATH es el directorio donde se encuentra la biblioteca
#nombre es el nombre de la imagen
#extension es la extensión de la imagen (PNG, JPG, etc).
#tempImage es la matriz tridimensional con valores entre 0 y 255.
