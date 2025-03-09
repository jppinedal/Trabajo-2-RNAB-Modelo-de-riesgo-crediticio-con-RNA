# Redes Neuronales y Algoritmos Bioinspirados

![logoUN](https://github.com/user-attachments/assets/6a75b35f-c2f7-425e-8a39-6d1384be3244)

# Modelos de riesgo de crédito con RNA
### Semestre 2024-2S

## Descripción del problema

El problema consiste en predecir la probabilidad de incumplimiento de pago de un crédito. La variable objetivo "loan_status" 
indica si un cliente incumplió con el pago de su crédito. La predicción de esta variable permite a las instituciones financieras 
minimizar riesgos y optimizar estrategias de concesión de créditos.

A continuación, se presenta el proceso de desarrollo de una Red Neuronal Artificial (RNA) que nos permite abordar este problema, 
ayudándonos a predecir el perfil de riesgo de un cliente dado, además de su puntaje crediticio basado en sus datos.

## Metodología

### 1. Delimitación del problema
- Predecir la probabilidad de incumplimiento de pago de un crédito utilizando un dataset de LendingClub.
- La variable objetivo "loan_status" indica si un cliente incumplió el pago.

### 2. Preprocesamiento de datos
- Se eliminaron créditos de tipo "JOINT" debido a su baja representatividad.
- Se removieron columnas con alta cantidad de datos nulos (>20%) y columnas irrelevantes.
- Se eliminaron columnas que no contribuían al análisis, como información personal no relevante y datos con valores únicos repetidos.
- Se normalizaron y codificaron las variables categóricas.
- El dataset final quedó con 32 variables, de las cuales se priorizaron las disponibles al momento de la solicitud del crédito. 
- Las variables finales utilizadas para el modelo son: 
  - 'annual_inc'
  - 'dti'
  - 'sub_grade'
  - 'open_acc'
  - 'total_acc'
  - 'inq_last_6mths'
  - 'delinq_2yrs'
  - 'int_rate'
  - 'pub_rec'
  - 'home_ownership'
  - 'purpose'
  - 'term'
  - 'verification_status'
  - 'installment'

### 3. Variable objetivo
- La variable "loan_status" se convirtió en binaria: 0 (cumplimiento) y 1 (incumplimiento).
- Se eliminaron las categorías indeterminadas, dejando solo las categorías de "Fully Paid" y "Charged Off" para el análisis.

### 4. Modelos trabajados
Se implementaron dos modelos de RNA secuenciales utilizando la librería Keras de Python.

#### Modelo 1
- RNA con dos capas ocultas (64 y 32 neuronas) y activación ReLU.
- Regularización mediante dropout (30%).
- Optimización con Adam y función de pérdida binary_crossentropy.
- Normalización con StandardScaler.

#### Modelo 2
- Se aplicó balanceo de clases, normalización con StandardScaler y reducción de dimensionalidad con PCA.
- Capas de Batch Normalization y Dropout para mitigar el sobreajuste.
- Optimización del umbral de clasificación basado en la métrica F1.

> [!NOTE]  
> [Enlace al informe completo](https://candy-monkey-1cd.notion.site/Trabajo-02-Modelos-de-riesgo-de-cr-dito-con-RNA-1897a8b98a1980bd96a6c5fe84c84960)

## Grupo 6
**Integrantes:**
- Valentina Ospina Narváez - vospinan@unal.edu.co
- Juan Camilo Torres Arboleda - jutorresar@unal.edu.co
- Juan Pablo Pineda Lopera - jppinedal@unal.edu.co
