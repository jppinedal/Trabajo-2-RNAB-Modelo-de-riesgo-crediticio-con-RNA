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
    model = load_model('src/modelo_credito.keras')
    with open('src/scaler.pkl', 'rb') as f:
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

# Campos de entrada con nombres mejorados y descripciones en USD
loan_amount = st.number_input("Monto del préstamo solicitado (USD)", min_value=1000, max_value=100000, value=10000, help="El monto del préstamo solicitado por el prestatario. Si el departamento de crédito lo reduce, se reflejará aquí.")
interest_rate = st.slider("Tasa de interés anual (%)", 1.0, 30.0, 12.5, help="Tasa de interés anual del préstamo.")
monthly_installment = st.number_input("Cuota mensual (USD)", min_value=50.0, max_value=3000.0, value=315.33, help="El pago mensual adeudado por el prestatario si el préstamo se origina.")
annual_income = st.number_input("Ingreso anual reportado (USD)", min_value=10000, max_value=500000, value=65000, help="El ingreso anual reportado por el prestatario durante el registro.")
debt_to_income_ratio = st.slider("Relación Deuda/Ingreso (DTI) (%)", 0.0, 100.0, 20.5, help="Relación calculada usando el total de pagos mensuales de deuda del prestatario sobre sus obligaciones totales de deuda, excluyendo hipotecas y el préstamo solicitado, dividido por el ingreso mensual reportado.")
delinquencies_past_2yrs = st.number_input("Número de delincuencias pasadas (2 años)", 0, 20, 0, help="El número de incidencias de delincuencia de 30+ días pasadas en el archivo de crédito del prestatario en los últimos 2 años.")
inquiries_last_6mths = st.number_input("Consultas de crédito recientes (6 meses)", 0, 20, 2, help="El número de consultas de crédito en los últimos 6 meses (excluyendo consultas de auto y hipoteca).")
open_credit_lines = st.number_input("Líneas de crédito abiertas", 0, 50, 8, help="El número de líneas de crédito abiertas en el archivo de crédito del prestatario.")
public_records = st.number_input("Registros públicos negativos", 0, 20, 0, help="Número de registros públicos derogatorios.")
revolving_balance = st.number_input("Saldo de crédito rotativo (USD)", 0, 100000, 15000, help="Total del saldo de crédito rotativo.")
revolving_utilization = st.slider("Utilización del crédito rotativo (%)", 0.0, 100.0, 55.5, help="Porcentaje de crédito rotativo utilizado por el prestatario en relación con todo el crédito disponible.")
total_credit_lines = st.number_input("Total de líneas de crédito", 0, 100, 12, help="El número total de líneas de crédito actualmente en el archivo de crédito del prestatario.")
employment_length = st.slider("Años de empleo", 0, 10, 5, help="Duración del empleo en años. Los valores posibles están entre 0 y 10, donde 0 significa menos de un año y 10 significa diez o más años.")

# Preparar datos para predicción
loan_data = {
    'loan_amnt': loan_amount,
    'int_rate': interest_rate,
    'installment': monthly_installment,
    'annual_inc': annual_income,
    'dti': debt_to_income_ratio,
    'delinq_2yrs': delinquencies_past_2yrs,
    'inq_last_6mths': inquiries_last_6mths,
    'open_acc': open_credit_lines,
    'pub_rec': public_records,
    'revol_bal': revolving_balance,
    'revol_util': revolving_utilization,
    'total_acc': total_credit_lines,
    'emp_length': employment_length
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