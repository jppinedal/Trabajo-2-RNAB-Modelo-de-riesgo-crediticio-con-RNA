import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import pickle
from pathlib import Path

# ----- Comienzo de la app ----- #
st.title("Redes Neuronales y algoritmos Bioinspirados")
st.subheader("2024-2S - Grupo 6 \n")

#current_dir = Path(__file__).parent # Directorio actual para hacer el despliegue
#file_path = current_dir / "img/logo.png"

# Configuración de la página
st.set_page_config(page_title="Scorecard de Riesgo Crediticio", layout="wide")
st.text("\n \n")
st.link_button("Conoce el informe técnico del proyecto", "https://candy-monkey-1cd.notion.site/Trabajo-02-Modelos-de-riesgo-de-cr-dito-con-RNA-1897a8b98a1980bd96a6c5fe84c84960?pvs=74", icon="📒", help="Haz clic en el enlace para ingresar al sitio web con el infórme técnico")
st.link_button("Repositorio de Github","https://github.com/jppinedal/Trabajo-2-RNAB-Modelo-de-riesgo-crediticio-con-RNA/tree/main", icon="📁", help="Haz clic en el enlace para conocer los notebook y el código de la app")

# Cargar modelo y scaler
try:
    model = load_model('src/loan_model.keras')
    with open('src/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open("src/pca.pkl", "rb") as f:
        pca = pickle.load(f)
    st.success("Modelo y scaler cargados exitosamente")
except Exception as e:
    st.error(f"Error al cargar el modelo o scaler: {e}")
    st.stop()


st.title(":blue[RISKO] :sunglasses: ")
st.title("Aplicación para Calcular Scorecard de Riesgo Financiero")
st.write("Ingrese los datos para realizar el cálculo.")

# Campos de entrada solicitados
annual_inc = st.number_input("Ingresos anuales (annual_inc)", min_value=0.01, value=60000.0, step=1000.0)

dti = st.slider("Relación Deuda-Ingreso (Debit-to-Income Ratio dti)", 0.0, 100.0, 25.0, help="Relación calculada usando el total de pagos mensuales de deuda del prestatario sobre sus obligaciones totales de deuda, dividido por el ingreso mensual reportado.")

# Mapeo de subgrados a valores numéricos
sub_grade_options = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}

# Escalas de 0.2 para los números
sub_grade_input = st.selectbox("Sub Grade", options=[f"{letter}{num}" for letter in sub_grade_options.keys() for num in range(1, 6)])

sub_grade_letter = sub_grade_input[0]
sub_grade_number = int(sub_grade_input[1])  

# Calcular el valor con escala de 0.2
sub_grade = sub_grade_options[sub_grade_letter] + (sub_grade_number - 1) * 0.2


open_acc = st.number_input("Líneas de crédito abiertas por el usuario (open_acc)", min_value=0.0, value=0.0, step=1.0)

total_acc = st.number_input("Líneas totales de crédito en el expediente del usuario (total_acc)", min_value=0.0, value=0.0, step=1.0)

inq_last_6mths = st.number_input("Número consultas en los últimos 6 meses (inq_last_6mths)", min_value=0.0, value=0.0, step=1.0)

delinq_2yrs = st.number_input("Número de incidencias de morosidad de más de 30 días en los últimos 2 años (delinq_2yrs))", min_value=0.0, value=0.0, step=1.0)

int_rate = st.slider("Tasa de interés (%)", 0.0, 100.0, 10.0, help="Tasa de interés anual del préstamo.")

pub_rec = st.number_input("Número de registros públicos derogatorios (pub_rec))", min_value=0.0, value=0.0, step=1.0, help="Cantidad de registros públicos negativos o adversos asociados a una persona o entidad en su historial financiero.")

# Lista desplegable para home_ownership
home_ownership_options = [
    "ANY", 
    "MORTGAGE",
    "NONE",
    "OTHER", 
    "OWN",
    "RENT"
]
home_ownership = st.selectbox("Tipo de propiedad de vivienda", options=home_ownership_options)

# Lista desplegable para purpose
purpose_options = [
    'car',
    'credit_card',
    'debt_consolidation',
    'educational',
    'home_improvement',
    'house',
    'major_purchase',
    'medical',
    'moving',
    'other',
    'renewable_energy',
    'small_business',
    'vacation',
    'wedding'
]
purpose = st.selectbox("Propósito del préstamo (purpose))", options=purpose_options)

# Plazo del préstamo en meses
# Opciones de plazo del préstamo
term_options = {"36 months": 0, "64 months": 1}

# Selector de plazo
term_selected = st.selectbox("Plazo del préstamo (term)", options=list(term_options.keys()))

# Convertir la selección a su valor correspondiente
term = term_options[term_selected]

# Verification status
verification_status_options = {"Not Verified": 0, "Source Verified": 0, "Verified": 2}

# Selector de estado de verificación
verification_status_selected = st.selectbox("Estado de verificación del usuario (verification_status)", options=list(verification_status_options.keys()))

# Convertir la selección a su valor correspondiente
verification_status = verification_status_options[verification_status_selected]


# Cuota mensual
monthly_installment = st.number_input("Cuota mensual (monthly_installment)", min_value=0.01, value=250.0, step=10.0, help="El pago mensual adeudado por el prestatario si el préstamo se origina.")


# Botón para calcular y mostrar los datos
if st.button("Calcular Riesgo"):
    # Preparamos datos en el orden exacto del ejemplo proporcionado
    # Orden: [annual_inc, dti, sub_grade, open_acc, total_acc, inq_last_6mths, delinq_2yrs, int_rate, pub_rec, 
    #         home_ownership (6 valores), purpose (14 valores), term, monthly_installment]
    
    # One-hot encoding para home_ownership
    home_ownership_values = [1 if option == home_ownership else 0 for option in home_ownership_options]
    
    # One-hot encoding para purpose
    purpose_values = [1 if option == purpose else 0 for option in purpose_options]
    
    # Crear el array en la estructura específica solicitada
    input_data = [
        annual_inc, 
        dti, 
        sub_grade, 
        open_acc, 
        total_acc, 
        inq_last_6mths, 
        delinq_2yrs, 
        int_rate, 
        pub_rec
    ] + home_ownership_values + purpose_values + [term, verification_status, monthly_installment]
    
    # Convertir a numpy array con la estructura exacta solicitada
    nuevo_dato = np.array([input_data])
    
    
    # Mostrar el tamaño y el contenido del array
    st.write(f"Forma del input: {nuevo_dato.shape}")
    #st.write(nuevo_dato)
    
    try:
        # Aplicar la normalización con el StandardScaler cargado
        nuevo_dato_scaled = scaler.transform(nuevo_dato)
        
        # Aplicar la transformación PCA
        nuevo_dato_pca = pca.transform(nuevo_dato_scaled)
        
        # Realizar la predicción con el modelo cargado
        probabilidad = model.predict(nuevo_dato_pca)
        umbral = 0.55  # Se puede ajustar según el problema
        prediccion = (probabilidad >= umbral).astype(int)  # Usar el mismo umbral ajustado

        # Mostrar resultados
        # Calcular probabilidad de incumplimiento en porcentaje
        probabilidad_incumplimiento = probabilidad[0][0] * 100  # Convertir a porcentaje

        # Gauge chart (Indicador principal del puntaje de incumplimiento)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilidad_incumplimiento,
            title={'text': "Probabilidad de Incumplimiento (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#ff4d4d' if probabilidad_incumplimiento > 70 else '#ffa600' if probabilidad_incumplimiento > 40 else '#00cc96'},
                'steps': [
                    {'range': [0, 40], 'color': '#00cc96'},   # (bajo riesgo)
                    {'range': [40, 70], 'color': '#ffa600'},  # Naranja  (riesgo medio)
                    {'range': [70, 100], 'color': '#ff4d4d'}  # Rojo (alto riesgo)
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': probabilidad_incumplimiento
                }
            }
        ))

        # Mostrar gráficos
        # Ajustar el tamaño del gráfico en Streamlit
        fig_gauge.update_layout(height=400, width=500)  # Tamaño ajustado

        # Mostrar resultados en Streamlit
        st.success("Predicción completada")
        st.subheader("Resultados:")
        st.write(f"Probabilidad de incumplimiento: {probabilidad_incumplimiento:.2f} %")
        st.write(f"Predicción: {'INCUMPLE' if prediccion[0][0] == 1 else 'CUMPLE'}")
        st.plotly_chart(fig_gauge)  # Mostrar el gráfico en Streamlit

    except Exception as e:
        st.error(f"Error durante el proceso: {e}")
        st.write("Detalles del error:", str(e))


# Pie de página
footer_placeholder = st.empty()
footer_html = """
    <style>
        .footer {
            text-align: center;
            margin-top: 50px;
            color: rgba(100, 100, 100, 0.7); /* Gris con opacidad */
            font-size: 14px;
        }
    </style>
    <div class="footer">
        2025 Universidad Nacional de Colombia Sede Medellín. <br>
        Redes Neuronales y Algortimos Bioinspirados. <br>
        Valentina Ospina Narváez, Juan Pablo Pineda Lopera, Juan Camilo Torres Arboleda
    </div>
"""
footer_placeholder.markdown(footer_html, unsafe_allow_html=True)
