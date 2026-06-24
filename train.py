"""
chatbot/model/train.py
======================
Pipeline de ML para clasificación de intents.

Matemáticas aplicadas:
  - TF-IDF: Term Frequency-Inverse Document Frequency
      TF(t,d)  = frecuencia del término t en documento d
      IDF(t)   = log(N / df(t))   — penaliza términos muy comunes
      TF-IDF   = TF × IDF         — peso en espacio vectorial R^|V|
  - SVM (Support Vector Machine):
      Busca el hiperplano w·x + b = 0 que maximiza el margen
      entre clases: max 2/||w||  sujeto a yi(w·xi+b) >= 1
      Con kernel RBF: K(x,z) = exp(-γ||x-z||²) para datos no lineales
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import re


# ── Preprocesamiento ─────────────────────────────────────────────────────────

def normalizar(texto: str) -> str:
    """
    Normalización lingüística básica:
      1. Minúsculas
      2. Eliminar acentos (opcional pero mejora el recall)
      3. Eliminar caracteres no alfanuméricos
      4. Strip de espacios
    """
    texto = texto.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
    texto = texto.lower().strip()
    # Normalizar acentos
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ü': 'u', 'ñ': 'n'
    }
    for orig, repl in reemplazos.items():
        texto = texto.replace(orig, repl)
    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


# ── Carga de datos ───────────────────────────────────────────────────────────

def cargar_datos(ruta: str):
    """
    Devuelve (X, y) donde:
      X = lista de frases normalizadas
      y = lista de etiquetas (tags)
    """
    with open(ruta, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X, y = [], []
    for intent in data['intents']:
        for patron in intent['patterns']:
            X.append(normalizar(patron))
            y.append(intent['tag'])

    print(f"  Dataset cargado: {len(X)} ejemplos, {len(set(y))} clases")
    for tag in sorted(set(y)):
        n = y.count(tag)
        print(f"    [{tag}]: {n} ejemplos")
    return X, y


# ── Construcción del pipeline ────────────────────────────────────────────────

def construir_pipeline():
    """
    Pipeline scikit-learn:
      1. TfidfVectorizer  → matriz dispersa R^(n × |V|)
         - analyzer='word': tokens a nivel de palabra
         - ngram_range=(1,2): unigramas + bigramas
         - sublinear_tf=True: TF = 1 + log(TF) para suavizar
         - min_df=1: incluir términos que aparecen al menos 1 vez
      2. SVC con kernel RBF
         - C: parámetro de regularización (penaliza errores)
         - gamma='scale': γ = 1/(n_features × Var(X))
         - probability=True: activa calibración de Platt para P(y|x)
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_features=5000,
        )),
        ('svm', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
        ))
    ])


# ── Entrenamiento y evaluación ───────────────────────────────────────────────

def entrenar(ruta_datos: str, ruta_modelo: str):
    print("\n=== ENTRENANDO CHATBOT (TF-IDF + SVM) ===\n")

    X, y = cargar_datos(ruta_datos)

    pipeline = construir_pipeline()

    # Validación cruzada estratificada (k=3 por el tamaño del dataset)
    # Estratificada → misma proporción de clases en cada fold
    print("\n  Validación cruzada (k=3, estratificada)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"  Accuracy por fold: {scores.round(3)}")
    print(f"  Media ± std: {scores.mean():.3f} ± {scores.std():.3f}")

    # Entrenamiento final con todos los datos
    print("\n  Entrenando con dataset completo...")
    pipeline.fit(X, y)

    # Reporte en train (para referencia — en producción usa CV)
    y_pred = pipeline.predict(X)
    print("\n  Reporte de clasificación (train):")
    print(classification_report(y, y_pred, zero_division=0))

    # Guardar modelo y metadatos
    Path(ruta_modelo).parent.mkdir(parents=True, exist_ok=True)

    # Cargar respuestas para guardarlas junto al modelo
    with open(ruta_datos, 'r', encoding='utf-8') as f:
        data = json.load(f)
    respuestas = {i['tag']: i['responses'] for i in data['intents']}

    modelo_completo = {
        'pipeline': pipeline,
        'respuestas': respuestas,
        'clases': list(set(y)),
    }

    with open(ruta_modelo, 'wb') as f:
        pickle.dump(modelo_completo, f)

    print(f"\n  Modelo guardado en: {ruta_modelo}")
    print("=== ENTRENAMIENTO COMPLETADO ===\n")
    return pipeline


# ── Función de inferencia ────────────────────────────────────────────────────

def predecir(pipeline, texto: str, umbral: float = 0.20):
    texto_norm = normalizar(texto)

    probs = pipeline.predict_proba([texto_norm])[0]
    clases = pipeline.classes_

    idx_max = np.argmax(probs)

    confianza = probs[idx_max]
    intent = clases[idx_max]

    print("\n----- PREDICCIÓN -----")
    print("Pregunta:", texto)
    print("Normalizada:", texto_norm)
    print("Intent predicho:", intent)
    print("Confianza:", confianza)
    print("----------------------\n")

    if confianza < umbral:
        intent = "desconocido"

    return intent, float(confianza)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE = Path(__file__).parent
    entrenar(
        ruta_datos=str(BASE / 'intents.json'),
        ruta_modelo=str(BASE / 'chatbot_model.pkl')
    )