from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb

app = FastAPI(title="Motor Predictivo Churn - XGBoost")

# 1. Cargar el modelo al levantar el servicio
modelo = xgb.XGBClassifier()
modelo.load_model("modelo_xgboost.json")

# 2. Definir el esquema del DTO que envía tu backend en .NET Core
class ClienteDTO(BaseModel):
    nro_cuotas_vencidas: int
    prima_bruta_anual: float
    meses_vigencia: int
    ano: int
    medio_pago: str  
    deducible: str   

@app.post("/api/predict")
def predict_churn(cliente: ClienteDTO):
    # Crear un diccionario base con ceros para las 19 variables
    datos_cliente = {
        'Nro. Cuotas vencidas': float(cliente.nro_cuotas_vencidas),
        'Prima Bruta Anual': float(cliente.prima_bruta_anual),
        'Meses Vigencia': float(cliente.meses_vigencia),
        'Año': float(cliente.ano),
        'Medio de Pago_PAT (debito cuenta corriente)': 0.0,
        'Medio de Pago_PAT (debito cuenta vista)': 0.0,
        'Deducible_Pendiente': 0.0,
        'Deducible_S/D': 0.0,
        'Deducible_UF 10': 0.0,
        'Deducible_UF 15': 0.0,
        'Deducible_UF 2.500': 0.0,
        'Deducible_UF 20': 0.0,
        'Deducible_UF 25': 0.0,
        'Deducible_UF 3': 0.0,
        'Deducible_UF 30': 0.0,
        'Deducible_UF 5': 0.0,
        'Deducible_UF 5.000': 0.0,
        'Deducible_UF 50': 0.0,
        'Deducible_UF 60': 0.0
    }
    
    # Activar la bandera (1.0) para el medio de pago recibido
    col_medio_pago = f"Medio de Pago_{cliente.medio_pago}"
    if col_medio_pago in datos_cliente:
        datos_cliente[col_medio_pago] = 1.0
        
    # Activar la bandera (1.0) para el deducible recibido
    col_deducible = f"Deducible_{cliente.deducible}"
    if col_deducible in datos_cliente:
        datos_cliente[col_deducible] = 1.0

    # Convertir a DataFrame manteniendo estrictamente el orden de las 19 columnas
    columnas_ordenadas = list(datos_cliente.keys())
    df_usuario = pd.DataFrame([datos_cliente], columns=columnas_ordenadas)
        
    probabilidad_fuga = modelo.predict_proba(df_usuario)[0][1] # Probabilidad de la clase 1
    
    # Retornar la respuesta estructurada a .NET Core
    return {
        "probabilidad_churn": round(float(probabilidad_fuga), 4),
        "riesgo_detectado": bool(probabilidad_fuga > 0.5),
        # Aquí posteriormente puedes mapear los factores TreeSHAP si los muestras en el Frontend
        "factores_clave": ["Morosidad Temprana" if cliente.nro_cuotas_vencidas > 0 else "Antigüedad Baja"]
    }