from pathlib import Path
from typing import List

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Motor Predictivo Churn - XGBoost")

# 1. Cargar el modelo una sola vez al levantar el servicio.
#    Ruta absoluta: funciona aunque uvicorn se lance desde otra carpeta.
RUTA_MODELO = Path(__file__).resolve().parent / "modelo_xgboost.json"
modelo = xgb.XGBClassifier()
modelo.load_model(RUTA_MODELO)

# 2. El orden de las columnas se toma del propio modelo, no de una lista escrita a mano:
#    si se reentrena con otras columnas, la API no se desincroniza en silencio.
COLUMNAS = modelo.get_booster().feature_names
INDICE_COLUMNA = {nombre: i for i, nombre in enumerate(COLUMNAS)}

COL_CUOTAS = "Nro. Cuotas vencidas"
COL_PRIMA = "Prima Bruta Anual"
COL_MESES = "Meses Vigencia"
COL_ANO = "Año"
PREFIJO_MEDIO_PAGO = "Medio de Pago_"
PREFIJO_DEDUCIBLE = "Deducible_"

_faltantes = [c for c in (COL_CUOTAS, COL_PRIMA, COL_MESES, COL_ANO) if c not in INDICE_COLUMNA]
if _faltantes:
    raise RuntimeError(f"El modelo no contiene las columnas esperadas: {_faltantes}")

UMBRAL_RIESGO = 0.5
MAX_CLIENTES_POR_LOTE = 1000


# 3. Esquemas de entrada/salida
class ClienteDTO(BaseModel):
    nro_cuotas_vencidas: int
    prima_bruta_anual: float
    meses_vigencia: int
    ano: int
    medio_pago: str
    deducible: str
    plan: str  # El modelo no lo usa; solo alimenta el mensaje explicativo.


class LoteClientesDTO(BaseModel):
    clientes: List[ClienteDTO] = Field(..., min_length=1, max_length=MAX_CLIENTES_POR_LOTE)


# 4. Función de explicabilidad usando reglas clásicas de negocio
def generar_mensaje_xai(cliente: ClienteDTO, prob: float):
    mensaje_1 = ""
    mensaje_2 = ""

    # Primer bloque: Evaluar el factor principal de riesgo de forma explícita
    if cliente.nro_cuotas_vencidas > 0:
        mensaje_1 = "Alta incidencia por morosidad en los pagos (" + str(cliente.nro_cuotas_vencidas) + " cuotas)"
    elif cliente.meses_vigencia < 12:
        mensaje_1 = "Riesgo asociado a la baja antigüedad de la póliza"
    elif cliente.medio_pago == "Aviso / Pago Manual":
        mensaje_1 = "El método de pago manual eleva la probabilidad de fuga"
    else:
        mensaje_1 = "El perfil presenta indicadores generales de riesgo"

    # Segundo bloque: Evaluar factores secundarios y de contexto
    if cliente.medio_pago == "Aviso / Pago Manual" and cliente.nro_cuotas_vencidas == 0:
        mensaje_2 = "y la falta de un pago automático debilita la retención."
    elif cliente.meses_vigencia >= 12 and cliente.nro_cuotas_vencidas > 0:
        mensaje_2 = "a pesar de tener un historial de permanencia previo."
    elif cliente.plan == "PT+RC":
        mensaje_2 = "sumado a las características de una cobertura básica (PT+RC)."
    elif prob > 0.8:
        mensaje_2 = "requiriendo una gestión de fidelización inmediata."
    else:
        mensaje_2 = "requiriendo monitoreo preventivo."

    return mensaje_1 + " " + mensaje_2


# 5. Conversión cliente -> vector numérico en el orden exacto del modelo
def vectorizar(cliente: ClienteDTO) -> np.ndarray:
    fila = np.zeros(len(COLUMNAS), dtype=np.float32)
    fila[INDICE_COLUMNA[COL_CUOTAS]] = cliente.nro_cuotas_vencidas
    fila[INDICE_COLUMNA[COL_PRIMA]] = cliente.prima_bruta_anual
    fila[INDICE_COLUMNA[COL_MESES]] = cliente.meses_vigencia
    fila[INDICE_COLUMNA[COL_ANO]] = cliente.ano

    # One-hot: si la categoría no existe en el modelo queda en la categoría base (todo 0),
    # igual que en el entrenamiento con drop_first.
    for columna in (PREFIJO_MEDIO_PAGO + cliente.medio_pago, PREFIJO_DEDUCIBLE + cliente.deducible):
        indice = INDICE_COLUMNA.get(columna)
        if indice is not None:
            fila[indice] = 1.0
    return fila


def construir_respuesta(cliente: ClienteDTO, probabilidad: float) -> dict:
    return {
        "probabilidad_churn": round(probabilidad, 4),
        "riesgo_detectado": probabilidad > UMBRAL_RIESGO,
        "factores_clave": [generar_mensaje_xai(cliente, probabilidad)],
    }


def predecir(clientes: List[ClienteDTO]) -> List[dict]:
    # Una sola llamada al modelo para todo el lote (XGBoost vectoriza internamente).
    matriz = np.vstack([vectorizar(c) for c in clientes])
    probabilidades = modelo.predict_proba(matriz)[:, 1]
    return [construir_respuesta(c, float(p)) for c, p in zip(clientes, probabilidades)]


# 6. Endpoints
@app.get("/health")
def health():
    """Liviano: sirve para 'despertar' el servicio en Render sin ejecutar el modelo."""
    return {"estado": "ok"}


@app.post("/api/predict")
def predict_churn(cliente: ClienteDTO):
    return predecir([cliente])[0]


@app.post("/api/predict/batch")
def predict_churn_batch(lote: LoteClientesDTO):
    """Evalúa varios clientes en una sola petición. Los resultados vienen en el mismo orden."""
    return {"resultados": predecir(lote.clientes)}
