# Detección de daño en imágenes satelitales post-Huracán mediante Deep Learning

### Link a la presentación del proyecto:
https://docs.google.com/presentation/d/1Y5RwZKD4ebCodnFGCTaJMX-61HYQy-HhzEoOLrGP6lM/edit?usp=sharing

![image](https://user-images.githubusercontent.com/77608824/112063858-ce492500-8b40-11eb-9087-be2ea1207cfc.png)

En este repo utilizamos las imágenes satelitales extraidas del siguiente link: https://drive.google.com/file/d/1BxmKyrxp5WrTqLzSLppiF-Gnd3KA6hF4/view?usp=sharing

Importamos el archivo a Google Colab, desde nuestro Google Drive, para poder trabajar sobre las imágenes utilizando un entorno de ejecución con GPU.

En la notebook presentada podemos encontrar lo siguiente:
  - Extracción de datos.
  - Observación de imágenes.
  - Confección de dataset y análisis de datos.
  - Procesamiento de datos.
  - Aumentación de datos.
  - Elaboración y entrenamiento de modelos de redes neuronales.
  - Evaluación de modelos.
  - Análisis de resultados.

## Resumen del Trabajo

Importamos el dataset en un archivo comprimido a Google Colab, desde nuestra unidad de Google Drive, para poder trabajar directamente sobre las imágenes.
Pasamos nuestros datos a un Pandas DataFrame para explorarlos y vemos la distribución espacial de nuestras imágenes en base a sus coordenadas satelitales (extraídas del nombre del archivo). De esta forma analizamos que las imágenes se encuentren balanceadas en cuanto a clase en nuestros sets de entrenamiento y validación.
Probamos que se lean bien las imágenes para evitar errores. Analizamos y procesamos en primer lugar nuestro set de entrenamiento y luego realizamos lo mismo para nuestro set de validación y de prueba.
  -	Para esto definimos una función que carga las imágenes como Array con OpenCV y las almacena en una lista.
  -	Mezclamos las imágenes ya que estaban ordenadas por clase, para no condicionar a la red por el posicionamiento de los datos cargados.
  -	Una vez mezclado separamos nuestra lista principal en 2 listas, una solo con los array de imágenes y otra con las etiquetas.
  -	Transformamos las listas en Arreglos de Numpy (Tensores) para que sea el formato válido de Keras.
  -	Pasamos nuestro arreglo de etiquetas a codificación one hot.

Luego aplicamos una aumentación de datos para nuestro set de entrenamiento y de validación, obteniendo una cantidad 4 veces mayor de datos para cada uno, 40000 imágenes y 8000 imágenes respectivamente.
  - Para esto realizamos un flip de las imágenes de izquierda a derecha, una rotación de +10º y una rotación de -10º sobre las imágenes originales.

Una vez analizados, procesados y aumentados nuestros datos pasamos a entrenar una red neuronal convolucional.

## Resultados

Antes de realizar la aumentación de datos entrenamos el modelo con los datos originales obteniendo resultados indeseados:
  - Se probó con 64 o 256 neuronas en la primera capa oculta obteniendo underfitting y todos los valores constantes de accuracy y loss a lo largo de las épocas.
  - Con 128 neuronas en la primera capa oculta tuvimos overfitting
  - Los resultados mejoraron un poco en validación aplicando relu en la capa fully connected y bajando a 32 la cantidad de neuronas en la primera capa oculta. De esta manera bajó mucho la cantidad de parámetros totales y disminuyó el overfitting. Total params: 1,865,250.

Para mejorar los resultados revisamos las salidas de las capas de nuestra red para analizar qué está haciendo la misma en su aprendizaje interno. Además, realizamos la aumentación de datos mencionada anteriormente, aumentamos la cantidad de épocas en el entrenamiento y utilizamos binary crossentropy (en lugar de categorical_crossentropy) como función de pérdida. Obtuvimos estos resultados:

![image](https://user-images.githubusercontent.com/77608824/112062604-e8820380-8b3e-11eb-82cd-b88318e66a77.png)

Para finalizar, mejoramos nuestro modelo logrando que los datos se ajusten al mismo utilizado un pequeño learning rate en el optimizador Adam a la hora de compilar el modelo. Es así que evaluamos resultados con los datos de validación y también con datos que nunca vio el modelo (test set) para asegurarnos que no haya overfitting:

![image](https://user-images.githubusercontent.com/77608824/112062726-136c5780-8b3f-11eb-882f-27a94d950d42.png)

![image](https://user-images.githubusercontent.com/77608824/112065816-1ae22f80-8b44-11eb-85b5-7e97798c5993.png)

Para concluir, obtuvimos una accuracy de 0.97 en nuestro test set luego de haber realizado una aumentación de datos de entrenamiento (x4), utilizado binary_crossentropy como función de pérdida y utilizado un pequeño learning rate en el optimizador Adam. Con estos resultados podemos concluir que el modelo se ajusta bien a nuestros datos y predice las clases con una gran precisión.

### Link del Notebook en Google Colab: 
https://drive.google.com/file/d/1cQUxzl3AYB3Do1ofaef2s6dsN0JZ2lsu/view?usp=sharing
