"""
retriever.py
============
Módulo RAG (Retrieval-Augmented Generation) simplificado.

Matemáticas:
  Dado un corpus de documentos d1, d2, ..., dn y una consulta q:

  1. Vectorizar con TF-IDF:
       vi = TF-IDF(di)  ∈ R^|V|
       vq = TF-IDF(q)   ∈ R^|V|

  2. Similitud coseno entre la consulta y cada documento:
       sim(q, di) = (vq · vi) / (||vq|| · ||vi||)  ∈ [0, 1]

     Nota: sklearn's TfidfVectorizer ya normaliza los vectores a norma 1
     (norm='l2' por defecto), así que el producto escalar equivale
     directamente a la similitud coseno:
       sim(q, di) = vq · vi   (si ||vq|| = ||vi|| = 1)

  3. Devolver el documento con mayor similitud si supera un umbral τ.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from train import normalizar


class Retriever:
    """
    Recuperador de documentos por similitud coseno TF-IDF.

    Atributos:
        documentos  — lista de dicts del knowledge_base.json
        vectorizer  — TfidfVectorizer ajustado al corpus
        matriz      — matriz TF-IDF del corpus, shape (n_docs, |V|)
    """

    def __init__(self, ruta_kb: str):
        # Cargar knowledge base
        with open(ruta_kb, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.documentos = data['documentos']

        # Construir corpus: concatenar titulo + contenido + usos para
        # maximizar el vocabulario útil de cada documento
        corpus = [
            normalizar(f"{d['titulo']} {d['contenido']} {d['usos']}")
            for d in self.documentos
        ]

        # Ajustar TF-IDF sobre el corpus
        # norm='l2' → vectores unitarios → producto escalar = similitud coseno
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm='l2',           # ||vi|| = 1  ← clave para la similitud coseno
            min_df=1,
        )
        # matriz shape: (n_documentos, |vocabulario|)
        self.matriz = self.vectorizer.fit_transform(corpus)

        print(f"[Retriever] KB cargada: {len(self.documentos)} documentos, "
              f"vocabulario={self.matriz.shape[1]} términos")

    def buscar(self, consulta: str, umbral: float = 0.15, top_k: int = 1):
        """
        Busca los documentos más similares a la consulta.

        Parámetros:
            consulta — texto libre del usuario
            umbral   — similitud mínima para considerar el resultado relevante
            top_k    — número de documentos a devolver

        Retorna:
            Lista de dicts con claves: documento, similitud
            Lista vacía si ningún documento supera el umbral.

        Matemáticas paso a paso:
            vq = vectorizer.transform([consulta])   → shape (1, |V|)
            scores = matriz · vq.T                  → shape (n_docs, 1)
              = [sim(q, d1), sim(q, d2), ..., sim(q, dn)]
        """
        consulta_norm = normalizar(consulta)

        # Vectorizar consulta: shape (1, |V|), también norma l2
        vq = self.vectorizer.transform([consulta_norm])

        # Producto escalar = similitud coseno (ambos vectores unitarios)
        # scores shape: (n_docs,)
        scores = (self.matriz * vq.T).toarray().flatten()

        # Índices ordenados de mayor a menor similitud
        indices_ordenados = np.argsort(scores)[::-1]

        resultados = []
        for idx in indices_ordenados[:top_k]:
            sim = float(scores[idx])
            if sim >= umbral:
                resultados.append({
                    'documento': self.documentos[idx],
                    'similitud': round(sim, 3)
                })

        # Log de depuración
        print(f"\n[Retriever] Consulta: '{consulta}'")
        print(f"[Retriever] Top resultado: "
              f"'{self.documentos[indices_ordenados[0]]['titulo']}' "
              f"(sim={scores[indices_ordenados[0]]:.3f})")
        print(f"[Retriever] Umbral {umbral} → "
              f"{'ENCONTRADO' if resultados else 'SIN RESULTADO'}")

        return resultados

    def formatear_respuesta(self, resultado: dict) -> str:
        """
        Formatea un documento recuperado como respuesta al usuario.
        """
        doc = resultado['documento']
        sim = resultado['similitud']

        respuesta = (
            f"Tenemos el producto **{doc['titulo']}** ({doc['marca']}): "
            f"{doc['contenido']} "
            f"Para más información o precio, llámanos al 925 23 34 54 "
            f"o escríbenos a almacen@olivillatres.com."
        )
        return respuesta