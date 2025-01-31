import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import pickle

# Configuración de la página
st.set_page_config(page_title="Scorecard de Riesgo Crediticio", layout="wide")

# Cargar modelo y scaler
try:
    model = load_model('modelo_credito.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error al cargar modelo o scaler: {e}")
    st.stop()

# Función para convertir probabilidad a puntaje FICO y categoría de riesgo
def probabilidad_a_fico(probabilidad):
    fico_score = 850 - (probabilidad * 550)  # Esto mapea 0 a 850 y 1 a 300
    if fico_score > 740:  # Excelente (por encima de 740 generalmente se considera excelente para hipotecas)
        categoria = "Excelente"
    elif fico_score > 670:  # Bueno (entre 670 y 739 es bueno para hipotecas)
        categoria = "Bueno"
    elif fico_score > 580:  # Aceptable (580-669 suele ser el mínimo para muchas hipotecas)
        categoria = "Aceptable"
    else:  # Pobre (por debajo de 580 es generalmente visto como alto riesgo)
        categoria = "Pobre"
    return fico_score, categoria

# Entrada de datos por parte del usuario
st.title("Datos Financieros para Calcular Scorecard de Riesgo Financiero")
st.write("Ingrese los datos para realizar el cálculo.")

# Campos de entrada
loan_amnt = st.number_input("Monto del préstamo", min_value=1000, max_value=100000, value=10000)
int_rate = st.slider("Tasa de interés (%)", 1.0, 30.0, 12.5)
installment = st.number_input("Cuota mensual", min_value=50.0, max_value=3000.0, value=315.33)
annual_inc = st.number_input("Ingreso anual", min_value=10000, max_value=500000, value=65000)
dti = st.slider("DTI (Debt-to-Income ratio)", 0.0, 100.0, 20.5)
delinq_2yrs = st.number_input("Número de delincuencias (2 años)", 0, 20, 0)
inq_last_6mths = st.number_input("Consultas últimos 6 meses", 0, 20, 2)
open_acc = st.number_input("Cuentas abiertas", 0, 50, 8)
pub_rec = st.number_input("Registros públicos", 0, 20, 0)
revol_bal = st.number_input("Balance revolvente", 0, 100000, 15000)
revol_util = st.slider("Utilización revolvente (%)", 0.0, 100.0, 55.5)
total_acc = st.number_input("Total de cuentas", 0, 100, 12)
emp_length = st.slider("Años de empleo", 0, 10, 5)

# Preparar datos para predicción
loan_data = {
    'loan_amnt': loan_amnt,
    'int_rate': int_rate,
    'installment': installment,
    'annual_inc': annual_inc,
    'dti': dti,
    'delinq_2yrs': delinq_2yrs,
    'inq_last_6mths': inq_last_6mths,
    'open_acc': open_acc,
    'pub_rec': pub_rec,
    'revol_bal': revol_bal,
    'revol_util': revol_util,
    'total_acc': total_acc,
    'emp_length': emp_length
}

if st.button("Calcular Riesgo"):
    input_df = pd.DataFrame([loan_data])
    
    # Escalar datos
    input_scaled = scaler.transform(input_df)
    
    # Realizar predicción
    probability = model.predict(input_scaled)[0][0]
    
    # Convertir probabilidad a puntaje FICO y categoría
    fico_score, categoria = probabilidad_a_fico(probability)

    # Datos para la scorecard
    data = pd.DataFrame({
        'Riesgo': ['Excelente', 'Bueno', 'Aceptable', 'Pobre'],
        'Porcentaje': [20, 30, 30, 20],  # Ajuste según tu análisis de datos o expectativas
        'PuntajePromedio': [750, 700, 625, 500]  # Valores aproximados
    })

    # Gráfico de barras: Distribución de categorías de riesgo con indicador
    fig_barras = go.Figure()

    # Barras de las categorías
    fig_barras.add_trace(go.Bar(
        x=data['Riesgo'],
        y=data['Porcentaje'],
        text=data['Porcentaje'],
        textposition='auto',
        marker=dict(color=['#00cc96', '#66ff66', '#ffa600', '#ff4d4d'])
    ))

    # Ajuste para mostrar el puntaje en el rango correcto de 'Porcentaje'
    puntaje_en_porcentaje = fico_score * max(data['Porcentaje']) / 850  # Escala el puntaje FICO al rango de porcentaje

    # Indicador visual del puntaje en las barras
    fig_barras.add_trace(go.Scatter(
        x=[categoria],
        y=[puntaje_en_porcentaje],
        mode='markers+text',
        marker=dict(size=12, color='white', symbol='diamond'),
        text=[f"Puntaje FICO: {fico_score:.1f}"],
        textposition='top center',
        name='Indicador Puntaje'
    ))

    fig_barras.update_layout(
        title='Distribución de Categorías de Riesgo con Indicador',
        xaxis_title='Categoría de Riesgo',
        yaxis_title='Porcentaje',
        template='plotly_white',
        yaxis_range=[0, max(data['Porcentaje']) * 1.1]  # Asegura que haya espacio para el marcador
    )

    # Gauge chart (Indicador principal del puntaje)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fico_score,
        title={'text': "Puntaje FICO"},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': '#00cc96' if fico_score > 740 else '#66ff66' if fico_score > 670 else '#ffa600' if fico_score > 580 else '#ff4d4d'},
            'steps': [
                {'range': [300, 580], 'color': '#ff4d4d'},
                {'range': [580, 670], 'color': '#ffa600'},
                {'range': [670, 740], 'color': '#66ff66'},
                {'range': [740, 850], 'color': '#00cc96'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': fico_score
            }
        }
    ))

    # Mostrar scorecard
    st.title("Scorecard de Riesgo Crediticio")
    st.write(f"Categoría de Riesgo: *{categoria}*")
    st.write(f"**Puntaje FICO:** {fico_score:.1f}")
    st.write("Este scorecard muestra la distribución del riesgo crediticio y el puntaje FICO basado en los datos ingresados.")

    # Mostrar gráficos
    st.plotly_chart(fig_barras, use_container_width=True)
    st.plotly_chart(fig_gauge, use_container_width=True)