"""
retriever.py
============
RAG ligero con TF-IDF + similitud coseno. Sin sentence-transformers.
Busca en knowledge_base.json por keywords y similitud de texto.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from train import normalizar


class Retriever:
    """
    Recuperador de documentos por similitud coseno TF-IDF.

    Construye una matriz TF-IDF del corpus de documentos al iniciar,
    luego para cada consulta calcula la similitud coseno contra todos
    los documentos y devuelve el más cercano si supera el umbral.
    """

    def __init__(self, ruta_kb: str):
        with open(ruta_kb, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.documentos = data['documentos']

        # Corpus: concatenar todos los campos relevantes de cada documento
        corpus = [
            normalizar(
                f"{d['titulo']} {d['contenido']} {d['usos']} {d.get('marca', '')}"
            )
            for d in self.documentos
        ]

        # TF-IDF sobre el corpus con n-gramas de palabras (1,2)
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.matriz = self.vectorizer.fit_transform(corpus)

        print(f"[Retriever] KB cargada: {len(self.documentos)} documentos, "
              f"vocabulario {len(self.vectorizer.vocabulary_)} términos")

    def buscar(self, consulta: str, umbral: float = 0.10, top_k: int = 1):
        """
        Devuelve los documentos más similares a la consulta.
        Umbral más bajo que con embeddings porque TF-IDF da scores menores.
        """
        consulta_norm = normalizar(consulta)
        vq = self.vectorizer.transform([consulta_norm])
        scores = cosine_similarity(vq, self.matriz)[0]

        indices_ordenados = np.argsort(scores)[::-1]
        mejor_idx = indices_ordenados[0]
        mejor_sim = float(scores[mejor_idx])

        print(f"[Retriever] '{consulta}' → "
              f"'{self.documentos[mejor_idx]['titulo']}' (sim={mejor_sim:.3f})")

        resultados = []
        for idx in indices_ordenados[:top_k]:
            sim = float(scores[idx])
            if sim >= umbral:
                resultados.append({
                    'documento': self.documentos[idx],
                    'similitud': round(sim, 3)
                })

        return resultados

    def formatear_respuesta(self, resultado: dict) -> str:
        doc = resultado['documento']
        return (
            f"Tenemos el producto **{doc['titulo']}** ({doc['marca']}): "
            f"{doc['contenido']} "
            f"Para más información o precio, llámanos al 925 23 34 54 "
            f"o escríbenos a almacen@olivillatres.com."
        )