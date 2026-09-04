### Proyecto de Machine Learning : Inteligencia Musical y Predicción de popularidad en Spotify
## 1. Problema del negocio
La compañia de spotify busca predecir la popularidad de canciones y poder desarrollar 
un algoritmo de inteligencia musical, por lo tanto hace entrega del dataset cargado(puro chamuyo chiques)
para determinar los factores de éxito de la industria.

## 2. Objetivos del proyecto
# **Objetivo General:** Desarrollar un modelo predictivo que clasifique la popularidad de las canciones según distintas variables.
# **Objetivos especificos:**  
  Identificar variables con alta correlacion con la variable objetivo "popularity"
  Identificar sesgos dentro del dataset
  Limpiar y transformar los datos musicales resolviendo problemas de alta cardinalidad
  Garantizar la integridad de los datos mediante validaciones de seguridad

## 3. Definiciones de KPIs
Para medir el exito del proyecto se definen los siguientes KPIs
*  **que metrica priorizamos chiques
*  **Reudccion de dimensionalidad:** Disminuir las variables de alta cardinalidad

## 4. Fuente de datos
Se utiliza la fuente de datos proveniente de la empresa **Spotify Tracks Dataset**
en formato .csv, se promueve su uso medienta la conexion con el repositorio en github mediante el url y la carga de datos con pandas


## 5. Metodologia CRISP-DM
La metodologia cuenta con seis fases principales que nos permitieron desarrollar el proyecto:
* **Comprension del negocio**
* **Comprension de los datos**
* **Preparacion de los datos**
* **Modelado**
* **Evaluacion**
* **Despliegue**

## 6. Analisis exploratorio
(aqui copie y pegue lo que puse en el notebook)
Para proteger la privacidad de los datos se ocupa la funcion SHA-256 para encriptarlos y mantenerlos anonimos,
esto ayuda a mantener la integridad de los datos de llegada para verificar que no esten corrompidos,
asegurando el cumplimiento da ley de proteccion de datos 21.719.\
El dataset presenta cierta de cantidad de datos con poca relevancia estadistica o duplicados, 
como se comprueba al buscar filas que tengan la columna "track_id" y "track_genre" con los mismos datos, se opta eliminarlos para mitigar sesgo innecesario, no obstante,
se mantienen filas que tienen los mismos datos a excepcion del genero, debido a la importancia del ultimo en un modelo que busca predecir la popularidad,
a consciencia del sesgo que llegara a producir dentro del algoritmo(ya que no hemos encontrado forma optima de tratarlo gemini activate).\
Se opta de la eliminacion de variables como 'track_id', 'track_name', 'album_name', 'energy' y 'Unnamed: 0', por la alta correlacion con la variable objetivo, 
ya que si fuera entrenado con esto se produciria sobreajuste(era ese verda, si, a ya).\
La variable 'artist' se busca mantener a pesar de su alta correlacion con el objetivo debido a la importancia del artista al momento de predecir popularidad,
por lo tanto se transforma los datos manteniendo a los artistas que cuentan con mas de 15 canciones, asegurando que el algoritmo aprenda los patrones de popularidad,
los demas que cuenten con una menor cantidad seran etiquetados como "Otros", esto a consciencia de bandas indie o artistas de un solo exito donde el algoritmo no aprendera
y su prescencia solo aportara ruido.
