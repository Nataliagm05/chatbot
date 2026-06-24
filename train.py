"""
chatbot/model/train.py
======================
Pipeline de ML para clasificación de intents.

Cambio respecto a la versión anterior:
  ANTES: TF-IDF (representación dispersa R^|V|) + SVM
  AHORA: Embeddings semánticos (representación densa R^384) + SVM

Matemáticas:
  - TF-IDF (versión anterior):
      Cada frase → vector disperso R^|V| basado en frecuencia de palabras exactas.
      "buenas tardes" y "buenas tardes a todos" → vectores distintos.

  - Embeddings semánticos (versión actual):
      Cada frase → vector denso R^384 que codifica el SIGNIFICADO.
      El modelo transforma la frase entera en un punto en R^384 tal que
      frases con significado similar quedan cerca (similitud coseno alta).
      "buenas tardes" y "buenas tardes a todos" → vectores casi idénticos.

  - SVM sobre embeddings:
      Igual que antes: busca el hiperplano w·x + b = 0 que maximiza el margen.
      Pero ahora x ∈ R^384 (denso) en vez de R^|V| (disperso).
      Con kernel lineal es suficiente porque los embeddings ya capturan
      la no-linealidad semántica. Con kernel RBF también funciona bien.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import re


# ── Modelo de embeddings (se carga una sola vez) ─────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2:
#   - 50 MB, funciona en CPU
#   - Entrenado en 50+ idiomas incluyendo español
#   - Produce vectores de dimensión 384
#   - Optimizado para similitud semántica entre frases cortas
EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# ── Preprocesamiento ─────────────────────────────────────────────────────────

def normalizar(texto: str) -> str:
    """
    Normalización lingüística básica.
    Se mantiene igual — la usa también retriever.py.
    Los embeddings no necesitan normalización estricta (el modelo
    ya maneja mayúsculas y acentos), pero la aplicamos igualmente
    para consistencia con el sistema de matching exacto.
    """
    texto = texto.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
    texto = texto.lower().strip()
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ü': 'u', 'ñ': 'n'
    }
    for orig, repl in reemplazos.items():
        texto = texto.replace(orig, repl)
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


# ── Vectorización con embeddings ─────────────────────────────────────────────

def vectorizar(frases: list) -> np.ndarray:
    """
    Convierte una lista de frases en una matriz de embeddings.

    Entrada:  lista de n frases
    Salida:   matriz numpy shape (n, 384)

    El modelo codifica cada frase como un vector denso de 384 dimensiones
    que representa su significado semántico. Frases similares en significado
    producen vectores cercanos en el espacio R^384 (similitud coseno alta).
    """
    return EMBEDDING_MODEL.encode(frases, show_progress_bar=False)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def entrenar(ruta_datos: str, ruta_modelo: str):
    print("\n=== ENTRENANDO CHATBOT (Embeddings + SVM) ===\n")

    X_texto, y = cargar_datos(ruta_datos)

    # Vectorizar frases de entrenamiento → matriz (n_ejemplos, 384)
    print("\n  Vectorizando frases con embeddings semánticos...")
    X = vectorizar(X_texto)
    print(f"  Matriz de embeddings: shape={X.shape}")

    # SVM directamente sobre los embeddings (sin Pipeline de sklearn
    # porque el vectorizador ya no es un transformador de sklearn)
    # Kernel RBF sigue siendo buena opción sobre embeddings densos
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',      # γ = 1/(384 × Var(X))
        probability=True,   # calibración de Platt para P(y|x)
        class_weight='balanced',
    )

    # Validación cruzada estratificada
    print("\n  Validación cruzada (k=3, estratificada)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')
    print(f"  Accuracy por fold: {scores.round(3)}")
    print(f"  Media ± std: {scores.mean():.3f} ± {scores.std():.3f}")

    # Entrenamiento final con todos los datos
    print("\n  Entrenando con dataset completo...")
    svm.fit(X, y)

    # Reporte en train
    y_pred = svm.predict(X)
    print("\n  Reporte de clasificación (train):")
    print(classification_report(y, y_pred, zero_division=0))

    # Guardar modelo
    Path(ruta_modelo).parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_datos, 'r', encoding='utf-8') as f:
        data = json.load(f)
    respuestas = {i['tag']: i['responses'] for i in data['intents']}

    modelo_completo = {
        'svm':       svm,          # ← ahora guardamos svm directamente, no pipeline
        'respuestas': respuestas,
        'clases':    list(set(y)),
    }

    with open(ruta_modelo, 'wb') as f:
        pickle.dump(modelo_completo, f)

    print(f"\n  Modelo guardado en: {ruta_modelo}")
    print("=== ENTRENAMIENTO COMPLETADO ===\n")
    return svm


# ── Inferencia ────────────────────────────────────────────────────────────────

def predecir(svm, texto: str, umbral: float = 0.35):
    """
    Predice el intent de un texto usando el SVM con embeddings.

    Diferencia clave respecto a la versión TF-IDF:
      Antes: pipeline.predict_proba([texto_normalizado])
             → el pipeline vectorizaba con TF-IDF internamente

      Ahora: vectorizamos con el modelo de embeddings primero,
             luego el SVM predice sobre ese vector denso.

    El umbral 0.35 funciona mejor ahora porque los embeddings
    producen probabilidades más discriminativas que TF-IDF.
    """
    texto_norm = normalizar(texto)

    # Vectorizar la consulta → shape (1, 384)
    x = vectorizar([texto_norm])

    # Probabilidades por clase
    probs  = svm.predict_proba(x)[0]
    clases = svm.classes_

    idx_max   = np.argmax(probs)
    confianza = probs[idx_max]
    intent    = clases[idx_max]

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