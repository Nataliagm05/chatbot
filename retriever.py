"""
retriever.py
============
Módulo RAG (Retrieval-Augmented Generation) con Embeddings Semánticos Locales.

Matemáticas:
  Dado un corpus de documentos d1, d2, ..., dn y una consulta q:
  1. Vectorizar conceptualmente con Sentence-Transformers (Modelo local gratis):
       vi = Model(di)  ∈ R^384
       vq = Model(q)   ∈ R^384
  2. El modelo ya entrega vectores normalizados (norma l2), por lo que
     el producto escalar (np.dot) equivale directamente a la similitud coseno.
  3. Entiende sinónimos y conceptos sin necesidad de coincidencia exacta de palabras.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from train import normalizar


class Retriever:
    """
    Recuperador de documentos por similitud coseno usando Embeddings Semánticos.

    Atributos:
        documentos  — lista de dicts del knowledge_base.json
        model       — Modelo de lenguaje SentenceTransformer cargado localmente
        matriz      — matriz de embeddings del corpus, shape (n_docs, 384)
    """

    def __init__(self, ruta_kb: str):
        # Cargar knowledge base
        with open(ruta_kb, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.documentos = data['documentos']

        # Construir corpus: concatenar campos importantes
        corpus = [
            normalizar(f"{d['titulo']} {d['contenido']} {d['usos']} {d.get('marca', '')}")
            for d in self.documentos
        ]

        print("[Retriever] Cargando modelo de lenguaje semántico local...")
        # Este modelo es gratuito, ligero (120MB), corre rápido en CPU y es excelente en español
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generar la matriz de embeddings conceptuales (se normalizan automáticamente)
        self.matriz = self.model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)

        print(f"[Retriever] KB cargada con Embeddings: {len(self.documentos)} documentos. "
              f"Matriz conceptual lista: shape {self.matriz.shape}")

    def buscar(self, consulta: str, umbral: float = 0.35, top_k: int = 1):
        """
        Busca los documentos conceptualmente más similares a la consulta.

        Parámetros:
            consulta — texto libre del usuario
            umbral   — similitud mínima (los embeddings requieren un umbral más alto, ~0.35 o 0.40)
            top_k    — número de documentos a devolver
        """
        consulta_norm = normalizar(consulta)

        # Generar embedding de la consulta del usuario (vector unitario)
        vq = self.model.encode([consulta_norm], convert_to_numpy=True, normalize_embeddings=True)

        # Producto escalar = similitud coseno
        # vq[0] extrae el vector 1D. El producto matriz-vector da las puntuaciones de cada doc.
        scores = np.dot(self.matriz, vq[0])

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
        print(f"\n[Retriever Semántico] Consulta: '{consulta}'")
        print(f"[Retriever Semántico] Top resultado conceptual: "
              f"'{self.documentos[indices_ordenados[0]]['titulo']}' "
              f"(sim={scores[indices_ordenados[0]]:.3f})")
        print(f"[Retriever Semántico] Umbral {umbral} → "
              f"{'ENCONTRADO' if resultados else 'SIN RESULTADO'}")

        return resultados

    def formatear_respuesta(self, resultado: dict) -> str:
        """
        Formatea un documento recuperado como respuesta al usuario.
        """
        doc = resultado['documento']
        respuesta = (
            f"Tenemos el producto **{doc['titulo']}** ({doc['marca']}): "
            f"{doc['contenido']} "
            f"Para más información o precio, llámanos al 925 23 34 54 "
            f"o escríbenos a almacen@olivillatres.com."
        )
        return respuesta
