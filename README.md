### Proyecto de Machine Learning : Inteligencia Musical y Predicción de popularidad en Spotify
## 1. Problema del negocio
La compañia de spotify busca predecir la popularidad de canciones y poder desarrollar 
un algoritmo de inteligencia musical, por lo tanto hace entrega del dataset cargado(puro chamuyo chiques)
para determinar los factores de éxito de la industria.

## 2. Objetivos del proyecto
**Objetivo General:** Desarrollar un modelo de aprendizaje que permita predecir, con una buena precisión, la popularidad de una canción. Además, se pretende encontrar patrones que permitan entender mejor las preferencias generales del público.
**Objetivos específicos:**  
-   Identificar variables con alta correlación con la variable objetivo "popularity".
-   Identificar sesgos dentro del dataset.
-   Limpiar y transformar los datos musicales resolviendo problemas de alta cardinalidad.
-   Garantizar la integridad de los datos mediante validaciones de seguridad.
-   Generar pipeline de preprocesamiento robusto y reproducible para futuros modelos de aprendizaje.

## 3. Definiciones de KPIs
Para medir el exito del proyecto se definen los siguientes KPIs
*  **que metrica priorizamos chiques
*  **Reudccion de dimensionalidad:** Disminuir las variables de alta cardinalidad

## 4. Fuente de datos
Se utiliza como fuente de datos un dataset guardado en un archivo .csv, que cuenta con información proveniente de la empresa **Spotify**. Se hace uso de este a través de una conexión con github mediante el url del repositorio, y se carga dentro del notebook utilizando la librería Pandas de Python.

## 5. Metodología CRISP-DM
La metodologia cuenta con seis fases principales que nos permitieron desarrollar el proyecto:
* **Comprensión del negocio**
  En la industria musical digital se vuelve necesario comprender los factores que impulsan el éxito comercial y la recepción de la audiencia, no sólo para los artistas, sino también para las propias plataformas. Es por lo anterior que se define como objetivo el desarrollo de un modelo analítico y predictivo que sea capaz de procesar la información referente a las canciones de Spotify para estimar su grado de éxito comercial, definiendo un flujo de trabajo que se centre en la estructura que facilite la reproducibilidad del entorno.
  
* **Comprensión de los datos**
  Todo el análisis es basado en el dataset presente en este repositorio de nombre 'Spotify_Tracks_Dataset.csv', el cuál contiene un volumen inicial de 114.000 registros, cada uno correspondiente a pistas musicales presentes en la plataforma. Cada registro cuenta con variables cuantitativas de audio así como datos descriptivos(artistas, álbumes, géneros).

  Durante la fase de exploración se identificaron dos dilemas significativos que requirieron análisis más detallado:
  - *Duplicidad por género:* Cuando una canción está asociada a más de un género, existían múltiples filas para esa canción en particular, lo que inflaba la cantidad de filas totales de 89740 canciones únicas a 114.000 filas de registros.
  - *Comportamiento de la variable objetivo:* Se observó que la variable objetivo para este proyecto('popularity') cuenta con una alta concentración en torno al valor 0, equivalente a aproximadamente un 14% de los registros.
* **Preparación de los datos**
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
