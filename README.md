### Proyecto de Machine Learning : Inteligencia Musical y Predicción de popularidad en Spotify
# 1. Descripción del Problema de Negocio

La industria de la música en streaming se caracteriza por un volumen masivo y continuo de lanzamientos, donde plataformas líderes como Spotify incorporan diariamente decenas de miles de canciones a su catálogo global. En este entorno hipercompetitivo, la curaduría editorial de listas de reproducción (*playlists*), el diseño de recomendaciones personalizadas y el descubrimiento temprano de talentos (*Artist & Repertoire / scouting*) constituyen procesos estratégicos críticos para sostener la retención y el compromiso (*engagement*) de los suscriptores. Recomendar temas que no conectan con la audiencia eleva la tasa de salto (*skip rate*) y deteriora la experiencia del usuario, mientras que no detectar a tiempo los nuevos éxitos representa una pérdida significativa de oportunidades comerciales frente a la competencia.

Frente a este escenario, la compañía requiere desarrollar una solución analítica de **inteligencia musical** sustentada en modelos de Machine Learning, capaz de estimar la popularidad de las canciones a partir de sus atributos acústicos intrínsecos (tales como bailabilidad, energía, sonoridad, valencia emocional o nivel de instrumentalidad) y sus metadatos de producción (género musical, artista y colaboraciones). La capacidad de anticipar el potencial de aceptación de una pista permite:

1. **Optimizar la selección de canciones para listas oficiales:** Priorizar canciones con alta probabilidad de éxito en listas oficiales y recomendaciones personalizadas, maximizando el tiempo de escucha en la plataforma.
2. **Potenciar el descubrimiento temprano de talentos:** Identificar canciones con potencial comercial provenientes de artistas emergentes o sellos independientes antes de su explosión orgánica en el mercado.
3. **Reducir el costo de oportunidad:** Disminuir la exposición de lanzamientos de bajo impacto en los sistemas de recomendación, evitando la fatiga auditiva del usuario.

El proyecto se desarrolla bajo el estándar de la metodología **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), iniciando con una fase rigurosa de comprensión del negocio y exploración de datos. Asimismo, la solución integra desde su diseño inicial el cumplimiento de estándares de gobernanza y privacidad de la información (conforme a la Ley N° 21.719 en Chile y el RGPD/GDPR europeo), junto con una auditoría activa de equidad algorítmica orientada a mitigar sesgos históricos que tiendan a invisibilizar a géneros y artistas minoritarios en favor de estilos masivos.

---

# 2. Objetivos del Proyecto

### 2.1. Objetivo General
Desarrollar un pipeline integral de preparación de datos y una solución analítica de Machine Learning bajo la metodología CRISP-DM, que permita predecir la popularidad de canciones en Spotify a partir de sus atributos acústicos y metadatos contextuales, identificando los factores determinantes del éxito musical para fundamentar la toma de decisiones en curaduría editorial y scouting artístico, garantizando la integridad de los datos, la reproducibilidad del proceso y la equidad algorítmica.

---

### 2.2. Objetivos Específicos

* **Auditar la procedencia, gobernanza e integridad técnica del dataset:** Implementar validaciones criptográficas mediante funciones hash (SHA-256) para certificar la inalterabilidad de los datos de entrada, garantizando el cumplimiento de los estándares éticos y normativos vigentes de protección de datos (Ley N° 21.719 y GDPR).
* **Ejecutar un análisis exploratorio de datos (EDA) y control de calidad:** Cuantificar y subsanar anomalías en el catálogo (gestión de duplicados por multiplicidad de géneros por pista, registros nulos e inconsistencias lógicas de rango) y evaluar distribuciones univariadas, justificando estadísticamente el criterio de retención o tratamiento de valores atípicos (*outliers*) en variables acústicas continuas.
* **Determinar correlaciones y dependencias multivariadas con la variable objetivo:** Medir la fuerza de asociación lineal y no lineal (coeficientes de Spearman y Razón de Correlación $\eta$ en atributos cualitativos) frente a `popularity`, identificando las características con verdadero poder explicativo y descartando identificadores o variables redundantes para prevenir sobreajuste (*overfitting*) y fuga de datos (*data leakage*).
* **Auditar y evaluar sesgos algorítmicos y de representatividad (Fairness):** Detectar la presencia de sesgos de selección e históricos dentro del catálogo musical (tales como la sobrerrepresentación de estilos comerciales en detrimento de escenas independientes), evaluando el impacto ético del modelo mediante métricas de justicia algorítmica como la Tasa de Impacto Dispar (*Disparate Impact Ratio*) bajo la regla del 80%.
* **Diseñar e implementar ingeniería de características vectorizada:** Generar nuevos predictores basados en interacciones acústicas (p. ej., combinación aditiva de valencia e instrumentalidad, segregación de artista principal y detección binaria de colaboraciones), mitigando el impacto de la alta cardinalidad mediante técnicas avanzadas de transformación y codificación (*Target Encoding* y *One-Hot Encoding*).
* **Construir un pipeline de preprocesamiento desacoplado y reproducible:** Estructurar un flujo modular mediante `ColumnTransformer` y separación estricta de particiones de entrenamiento y prueba (*Train/Test split*), garantizando la reproducibilidad metodológica y la preparación de los datos para la futura fase de modelamiento predictivo (regresión o clasificación).

# 3. Definición de KPIs del Proyecto

Para evaluar el desempeño integral y la viabilidad del proyecto, se definen indicadores clave de rendimiento (KPIs) divididos en dos niveles complementarios: **KPIs de Negocio** (impacto operacional y valor estratégico en la plataforma) y **KPIs Técnicos de Machine Learning** (precisión matemática, capacidad predictiva y justicia algorítmica).

---

### 3.1. KPIs de Negocio
Miden el impacto directo de la solución sobre la experiencia del usuario y la gestión del catálogo de Spotify:

**Tasa de Detección Temprana de Éxitos Independientes (*Breakout Discovery Rate*):**
   * **Definición:** Porcentaje de canciones producidas por artistas emergentes o de sellos independientes (con catálogos históricos reducidos) con alto potencial de éxito que el sistema identifica con precisión.
   * **Meta Cuantificable:** Lograr que el modelo identifique correctamente al menos el **70% de las canciones independientes con potencial de éxito** (definidas con un umbral de popularidad $\ge 60$ puntos), facilitando su inclusión temprana en playlists destacadas y optimizando la labor de búsqueda de nuevos talentos (*scouting* artístico).

---

### 3.2. KPIs Técnicos de Machine Learning
Evalúan la calidad estadística de las predicciones y la equidad del pipeline analítico:

**Error Absoluto Medio (*Mean Absolute Error* - MAE):**
   * **Definición:** Métrica continua que cuantifica el promedio de las diferencias absolutas entre el puntaje de popularidad real de la canción y el estimado por el modelo.
   * **Meta Cuantificable:** Obtener un **$\text{MAE} < 5$ puntos de popularidad** (en la escala nativa de 0 a 100) evaluado sobre el conjunto de prueba desacoplado (*test set*), asegurando un margen de error estrecho y confiable para la toma de decisiones.


---


# 4. Descripción de las Fuentes de Datos y Entorno Tecnológico

### 4.1. Ficha Técnica y Estructura del Dataset
La base de información empleada corresponde al conjunto de datos **Spotify Tracks Dataset**, derivado de la API oficial de Spotify (*Spotify Web API*). El dataset crudo inicial cuenta con **114.000 registros** y **21 atributos estructurados**, que combinan identificadores, metadatos artísticos y variables de procesamiento de señales de audio:

* **Identificadores y Metadatos:** `track_id` (identificador único alfanumérico de Spotify), `artists` (artista o artistas intérpretes, separados por punto y coma), `album_name` (álbum de publicación), `track_name` (título de la canción) y `track_genre` (clasificación estilística con 114 géneros musicales únicos).
* **Variable Objetivo (*Target*):** `popularity` (variable numérica entera que oscila entre 0 y 100, calculada dinámicamente por la plataforma según el volumen y recencia de reproducciones).
* **Variables de Audio Intrínsecas:** Atributos acústicos normalizados entre $0.0$ y $1.0$ (`danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`), variables físicas (`loudness` en decibeles, `tempo` en BPM, `duration_ms` en milisegundos), y descriptores tonales (`key`, `mode`, `time_signature`).
* **Contenido Explícito:** `explicit` (indicador booleano que señala si la pista contiene letras explícitas).

---

### 4.2. Procedencia, Ingesta Remota e Integridad Criptográfica
Para asegurar la reproducibilidad de la solución sin depender de descargas manuales locales, el archivo de datos (`Spotify_Tracks_Dataset.csv`) se encuentra alojado en un repositorio público de **GitHub**. 

La ingesta se efectúa en memoria mediante una petición HTTP vía `requests` y decodificación binaria con `io.BytesIO`, parseada directamente mediante la librería `pandas` en Python. Con el fin de certificar la inalterabilidad de la fuente desde su origen, el pipeline ejecuta de forma obligatoria una auditoría criptográfica mediante la función hash **SHA-256**:

$$\text{Hash Verificado:} \quad \texttt{b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd}$$

Si el valor calculado sobre los bytes entrantes difiere del valor de control original, la ejecución se interrumpe de inmediato mediante excepciones controladas (`ValueError` / `RuntimeError`), garantizando que ningún dato corrompido o adulterado ingrese a la fase analítica.

---

### 4.3. Auditoría de Privacidad y Cumplimiento Normativo (PII Audit)
Se realizó una inspección sobre el contenido de la fuente para determinar la presencia de información de identificación personal (*Personally Identifiable Information* - PII). Se verificó que el conjunto de datos:
* **No contiene identificadores de usuarios:** No almacena RUT, direcciones de correo, identificadores de clientes, hábitos individuales de consumo ni registros de geolocalización de oyentes.
* **Tratamiento de figuras públicas:** Los nombres registrados corresponden a artistas y figuras de notoriedad pública dentro del ámbito comercial musical.
* **Cumplimiento legal:** El manejo de la información respeta los estándares vigentes de la regulación nacional de protección de datos personales de Chile (**Ley N° 21.719**) y los principios de minimización del dato del **RGPD/GDPR europeo**.

---

### 4.4. Justificación de Herramientas Colaborativas y Ecosistema Técnico
En conformidad con las directrices de la metodología CRISP-DM, se adoptó el siguiente entorno tecnológico colaborativo:

1. **GitHub (Control de Versiones y Distribución):** Utilizado como repositorio central para el versionamiento semántico del código fuente (`.ipynb`), el seguimiento de cambios (*commits*) entre integrantes del equipo y la distribución desacoplada de los datos crudos.
2. **Google Colab (Entorno de Cómputo Colaborativo en la Nube):** Seleccionado como entorno de desarrollo unificado para permitir la edición concurrente y la reproducibilidad exacta de dependencias analíticas en máquinas virtuales homogéneas, eliminando discrepancias operativas entre sistemas operativos locales.
3. **Ecosistema Científico de Python:**
   * `pandas` y `numpy`: Para la ingesta, manipulación estructurada y cálculo vectorial eficiente.
   * `matplotlib` y `seaborn`: Para la construcción de visualizaciones estadísticas (histogramas, boxplots y barras comparativas.
   * `scipy`: Para pruebas estadísticas inferenciales (Chi-cuadrado y cálculo de V de Cramér).
   * `scikit-learn` y `category_encoders`: Para el encapsulamiento del flujo de transformación mediante `ColumnTransformer` (con `OneHotEncoder`, `TargetEncoder` y escaladores numéricos), asegurando una separación estricta entre entrenamiento y prueba (*Train/Test split*) sin fuga de información (*data leakage*).

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
  Limpieza de nulos, eliminación de duplicados, Feature Engineering 
* **Modelado**
*   Aun por hacer
* **Evaluacion**
*   Aun por hacer
* **Despliegue**
*   Aun por hacer

## 6. Analisis exploratorio
Para proteger la privacidad de los datos se ocupa la función SHA-256 para encriptarlos y mantenerlos anónimos,
esto ayuda a mantener la integridad de los datos de llegada para verificar que no estén corrompidos,
asegurando el cumplimiento da ley de protección de datos 21.719.\
El dataset presenta cierta de cantidad de datos con poca relevancia estadística o duplicados, 
cómo se comprueba al buscar filas que tengan la columna "track_id" y "track_genre" con los mismos datos, se opta eliminarlos para mitigar sesgo innecesario, no obstante,
se mantienen filas que tienen los mismos datos a excepción del género, debido a la importancia del último en un modelo que busca predecir la popularidad,
a consciencia del sesgo que llegara a producir dentro del algoritmo.\
Se opta de la eliminacion de variables como 'track_id', 'track_name', 'album_name', 'energy' y 'Unnamed: 0', por la alta correlacion con la variable objetivo, 
ya que si fuera entrenado con esto se produciria sobreajuste(era ese verda, si, a ya).\
La variable 'artist' se busca mantener a pesar de su alta correlacion con el objetivo debido a la importancia del artista al momento de predecir popularidad,
por lo tanto se transforma los datos manteniendo a los artistas que cuentan con mas de 15 canciones, asegurando que el algoritmo aprenda los patrones de popularidad,
los demas que cuenten con una menor cantidad seran etiquetados como "Otros", esto a consciencia de bandas indie o artistas de un solo exito donde el algoritmo no aprendera
y su prescencia solo aportara ruido.
