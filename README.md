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

# 5. Preparación y Análisis Exploratorio de los Datos (EDA)

El análisis exploratorio de datos y el flujo de preparación se ejecutaron de manera secuencial e iterativa, garantizando que cada transformación estuviera técnicamente justificada por la evidencia estadística encontrada y blindada contra la fuga de datos (*data leakage*).

---

### 5.1. Auditoría de Calidad, Limpieza y Depuración Estructural
Antes de realizar inferencias estadísticas, se auditó la sanidad estructural del conjunto de datos (114.000 filas y 21 columnas iniciales):

1. **Gestión de Registros Nulos:**
   * **Hallazgo:** Se detectó exactamente **1 registro** con valores faltantes (`NaN`) en los metadatos textuales `artists`, `track_name` y `album_name` (asociado al identificador `track_id: 1kR4gIb7nGxHPI3D2ifs59`).
   * **Acción y Justificación:** Se aplicó eliminación por lista (*listwise deletion*). Al representar el 0.00087% del volumen total, su exclusión no introduce sesgo de selección muestral ni altera las distribuciones acústicas. Rellenarlo con categorías artificiales como *"Desconocido"* habría distorsionado la posterior codificación y agrupamiento por artista.
2. **Depuración de Índices Redundantes:**
   * Se descartó la columna `Unnamed: 0`, la cual correspondía a un índice residual heredado del almacenamiento serializado del CSV sin relevancia analítica ni predictiva.
3. **Conversión de Unidades Físicas:**
   * La variable `duration_ms` fue convertida a escala de segundos (`duration_s = duration_ms / 1000`) para mejorar la interpretabilidad operativa y la visualización de distribuciones temporales.
4. **El Dilema de los Duplicados por Multiplicidad de Género:**
   * **Hallazgo:** La auditoría arrojó 114.000 filas, pero solo **89.740 canciones únicas** por `track_id`. Se descubrió que Spotify registra una misma canción en múltiples filas para asignarle distintos géneros (`track_genre`), existiendo ~25.000 registros repetidos estructuralmente.
   * **Riesgo Técnico:** Mantener filas repetidas con idénticos atributos acústicos provocaría que una misma pista quede repartida entre los conjuntos de entrenamiento y prueba (*Train/Test split*), ocasionando **fuga de datos (*data leakage*)** y memorización espuria.
   * **Acción:** Se colapsó el dataset a registros únicos mediante `groupby('track_id').first()`, garantizando que cada unidad observacional sea tratada una sola vez en el pipeline analítico.

---

### 5.2. Análisis Univariado y Tratamiento de Valores Atípicos (*Outliers*)

1. **Comportamiento Distribucional del Target (`popularity`):**
   * **Concentración en Cero:** Aproximadamente el **14% de las canciones presentan una popularidad exactamente igual a 0**. Este fenómeno refleja la realidad del negocio: una gran masa de canciones de catálogo profundo o lanzamientos independientes que no logran tracción algorítmica ni reproducciones mínimas.
   * **Asimetría:** La distribución general está sesgada a la izquierda; la gran mayoría de las canciones se concentra en el rango de $[0, 50]$ puntos, con una cola reducida de éxitos comerciales que superan los 60-75 puntos.
2. **Inspección de Atributos Acústicos Continuos:**
   * Se inspeccionaron variables acotadas al rango $[0.0, 1.0]$ (`danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`) y variables continuas abiertas (`loudness`, `tempo`, `duration_s`).
   * **Valores Atípicos (*Outliers*):** Los diagramas de caja (*boxplots*) revelaron colas extendidas en `duration_s` (canciones muy largas o audiolibros/sonidos ambientales) y `loudness` (pistas acústicas o silencios extremos).
   * **Decisión Técnica:** **Se decidió conservar los valores atípicos** sin truncamiento ni eliminación indiscriminada. Estos registros representan expresiones sonoras legítimas de la diversidad musical de Spotify (música clásica, jazz experimental, pistas para dormir). Su tratamiento se delega a técnicas de transformación y escalamiento robusto en la fase de preprocesamiento para evitar distorsiones inducidas.

---

### 5.3. Análisis Bivariado, Multivariado y Dependencias Estadísticas

1. **Evaluación de Correlación Lineal (Spearman):**
   * Se calculó la matriz de correlación de Spearman entre los atributos de audio numéricos y `popularity`.
   * **Hallazgo Crítico:** La asociación lineal directa entre los descriptores acústicos individuales y la popularidad es notablemente débil: **ninguna variable numérica original supera $\vert{}r\vert{} = 0.12$** frente al target.
   * **Conclusión Estadística:** Los atributos acústicos por sí solos explican menos del 1.5% de la varianza del éxito musical. El éxito comercial en streaming no depende exclusivamente de cómo suena una pista, sino de factores contextuales como el género musical, el alcance del artista y las colaboraciones estratégicas.
2. **Asociación Categórica Multidimensional (V de Cramér / Razón $\eta$):**
   * Al categorizar la popularidad en quintiles (`popularity_cat`), la métrica **V de Cramér** evidenció una asociación sustancial con `track_genre` y con los artistas.
   * **Medianas por Género:** Los géneros comerciales como *pop*, *dance* y *latino* presentan medianas sistemáticas sobre los 60 puntos y una alta tasa de éxitos (*hit rate* con popularidad $> 75$), mientras que categorías como *black-metal*, *classical* o *study* se concentran cerca de la base del target.
3. **Contraste de Perfiles (Canciones Populares vs. No Populares):**
   * Al comparar las medias acústicas entre canciones populares ($> 75$) e impopulares, se observó que la mayoría de los descriptores son similares, excepto por dos métricas distintivas: **la acústica (`acousticness`) y la instrumentalidad (`instrumentalness`) descienden drásticamente en las canciones altamente exitosas**, confirmando que el público masivo prefiere producciones con alta presencia vocal y sintetizada.

---

### 5.4. Auditoría de Ética, Privacidad y Sesgos Algorítmicos

1. **Privacidad y Cumplimiento Normativo (Ley 21.719 y GDPR):**
   * Se comprobó la ausencia total de datos de identificación personal directa de usuarios (PII). La verificación criptográfica con **SHA-256** certifica la inalterabilidad de la fuente desde su repositorio en GitHub.
2. **Sesgo de Selección e Histórico:**
   * El dataset presenta un sesgo histórico de la industria: los algoritmos pasados y la industria radial han privilegiado la visibilidad de géneros masivos (*pop*, *reggaeton*), relegando la música independiente.
3. **Auditoría de Impacto Dispar (Regla del 80%):**
   * Si se evaluara un modelo predictivo optimizado únicamente por precisión matemática sin restricciones, este tendería a predecir popularidad baja para cualquier artista de géneros de nicho.
   * Se incorpora como restricción de gobernanza evaluar el **Ratio de Impacto Dispar (DIR)** entre géneros minoritarios y comerciales ($\text{DIR} \ge 0.80$), garantizando que el pipeline no discrimine sistemáticamente a la música emergente.

---

### 5.5. Ingeniería de Características (*Feature Engineering*) Vectorizada

Para superar la baja señal predictiva de las variables acústicas crudas y manejar la alta dimensionalidad, se implementaron transformaciones vectorizadas con NumPy y Pandas:

* **Interacción Acústica Aditiva (`valence_instrumentalness_sum`):** Se combinaron `valence` (positividad emocional) e `instrumentalness` (ausencia de voces). Mientras individualmente presentaban correlaciones marginales (0.04 y 0.09), su suma aritmética elevó la fuerza asociativa lineal a un **12% ($r \approx 0.12$)** con la popularidad.
* **Segregación del Artista Principal (`artista_principal`):** En pistas con múltiples cantantes (separados por punto y coma), se extrajo vectorialmente el primer intérprete (`artists.str.split(';').str[0]`), aislando al artista de mayor impacto comercial y reduciendo la dispersión textual.
* **Detección Binaria de Colaboración (`es_colaboracion`):** Se creó una bandera booleana (`1/0`) mediante detección de delimitadores en la cadena de artistas (`contains(';')`), capturando el efecto positivo que las colaboraciones (*feats*) generan en la visibilidad de Spotify.
* **Control de Cardinalidad de Géneros:** De los 114 géneros iniciales, se calcularon frecuencias relativas sobre las canciones desduplicadas; los géneros que representaban un 0.75% o menos del catálogo fueron agrupados bajo la etiqueta general `'otros'`, reduciendo la dimensionalidad a categorías representativas y mitigando el sobreajuste.
* **Descarte Justificado de Variables:** Se eliminaron identificadores y metadatos no generalizables (`track_id`, `track_name`, `album_name`) por su cardinalidad extrema que induciría memorización espuria (*overfitting*), así como variables acústicas redundantes cuya señal quedó capturada en las nuevas características.

---

### 5.6. Pipeline de Preprocesamiento Desacoplado
Para garantizar la reproducibilidad y prevenir la fuga de información entre particiones:

1. **Partición Estratificada (*Train/Test Split*):** Se separó el 80% de los datos para entrenamiento y el 20% para evaluación (`test_size=0.2`, `random_state=42`), estratificando por la variable de salida para mantener intacta la proporción de éxitos y temas impopulares en ambas muestras.
2. **Ensamblaje con `ColumnTransformer`:**
   * **Variables numéricas** (`speechiness`, `valence_instrumentalness_sum`, `es_colaboracion`): Flujo directo (`passthrough`).
   * **Variable categórica nominal** (`track_genre`): Codificación `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   * **Variable de alta cardinalidad** (`artista_principal`): Codificación supervisada mediante `TargetEncoder()`, aprendiendo el impacto del artista ajustado exclusivamente sobre el conjunto de entrenamiento (`X_train`).

```python
# Arquitectura del Pipeline de Transformación
preprocesador = ColumnTransformer(
    transformers=[
        ("num", "passthrough", cols_num),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_nom_ohe),
        ("te", TargetEncoder(), cols_nom_te)
    ],
    remainder="drop",
    verbose_feature_names_out=False
).set_output(transform="pandas")

X_train_procesado = preprocesador.fit_transform(X_train, y_train)
X_test_procesado = preprocesador.transform(X_test)
los demas que cuenten con una menor cantidad seran etiquetados como "Otros", esto a consciencia de bandas indie o artistas de un solo exito donde el algoritmo no aprendera
y su prescencia solo aportara ruido.
