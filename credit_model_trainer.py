import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
import pickle
import shap

# Rutas de los archivos
rutaOrigen = 'loan\loan.csv'
rutaModelo = 'modelo_credito.keras'
rutaScaler = 'scaler.pkl'

def load_and_preprocess_data(data):
    # Convertir loan_status a variable binaria (1 para default/charged off, 0 para otros)
    default_conditions = ['Default', 'Charged Off', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']
    df = data.copy()
    df['default'] = df['loan_status'].apply(lambda x: 1 if x in default_conditions else 0)

    # Seleccionar features relevantes
    features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
                'revol_bal', 'revol_util', 'total_acc', 'emp_length']

    # Preprocesar emp_length
    df['emp_length'] = df['emp_length'].fillna('0')
    df['emp_length'] = df['emp_length'].replace('10+ years', '10')
    df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)

    # Preparar X e y
    X = df[features].copy()
    y = df['default']

    # Imputar valores faltantes
    X = X.fillna(X.mean())

    return X, y

def create_model(input_dim):
    # Aquí defines tu arquitectura de modelo. Esto es solo un ejemplo:
    model = Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    """
    Entrena el modelo y guarda tanto el modelo como el scaler
    """
    # Cargar y preprocesar datos
    df = pd.read_csv(rutaOrigen, on_bad_lines='skip')
    X, y = load_and_preprocess_data(df)
    
    # Split y escalado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Crear y entrenar modelo
    model = create_model(X_train.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluar modelo
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo y scaler
    model.save(rutaModelo)  # Guardar el modelo en la ruta especificada
    with open(rutaScaler, 'wb') as f:
        pickle.dump(scaler, f)  # Guardar el scaler en la ruta especificada
    
    print(f"\nModelo guardado en: {rutaModelo}")
    print(f"Scaler guardado en: {rutaScaler}")
    
    return model, scaler

# Ejecutar la función para entrenar y guardar el modelo
if __name__ == "__main__":
    train_and_save_model()