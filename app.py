"""
app.py
======
Servidor Flask definitivo optimizado para PythonAnywhere.
Flujo híbrido con guardado automático de preguntas sin responder.
"""

import pickle
import random
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import sys

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import train
print(">>> TRAIN DESDE:", train.__file__)
print(">>> TIENE vectorizar:", hasattr(train, 'vectorizar'))

from train import normalizar, predecir
from retriever import Retriever          

from flask_cors import CORS
app = Flask(__name__)
# Permite que cualquier página web externa se conecte a este servidor Flask
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["POST", "GET", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})


# ── Cargar datos ─────────────────────────────────────────────────────────────

MODELO_PATH  = BASE / 'chatbot_model.pkl'
INTENTS_PATH = BASE / 'intents.json'
KB_PATH      = BASE / 'knowledge_base.json'    

with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

# Entrenar / recargar modelo SVM al arrancar
from train import entrenar
entrenar(
    ruta_datos=str(INTENTS_PATH),
    ruta_modelo=str(MODELO_PATH)
)

with open(MODELO_PATH, 'rb') as f:
    modelo_data = pickle.load(f)

svm        = modelo_data['svm']
respuestas = modelo_data['respuestas']

# Inicializar Retriever RAG
retriever = Retriever(str(KB_PATH))            


# ── Lógica de respuesta ───────────────────────────────────────────────────────

def obtener_respuesta(texto: str):
    texto_norm = normalizar(texto)

    # ── Nivel 1: coincidencia exacta ─────────────────────────────────────────
    for intent in intents_data["intents"]:
        patrones = [normalizar(p) for p in intent["patterns"]]
        if texto_norm in patrones:
            return {
                "intent":    intent["tag"],
                "nivel":     "exacto",
                "confianza": 1.0,
                "respuesta": random.choice(intent["responses"])
            }

    # ── Nivel 2: clasificador SVM ─────────────────────────────────────────────
    intent, confianza = predecir(svm, texto)

    if intent != 'desconocido' and intent in respuestas:
        return {
            'intent':    intent,
            'nivel':     'svm',
            'confianza': round(confianza, 3),
            'respuesta': random.choice(respuestas[intent])
        }

    # ── Nivel 3: RAG por similitud coseno o coincidencia conceptual ──────────
    resultados = retriever.buscar(texto, umbral=0.20)

    if resultados:
        mejor = resultados[0]
        return {
            'intent':    'rag_kb',
            'nivel':     'rag',
            'confianza': mejor['similitud'],
            'respuesta': retriever.formatear_respuesta(mejor)
        }

    # ── Nivel 4: fallback (Aprendizaje pasivo automático) ─────────────────────
    try:
        UNANSWERED_PATH = BASE / 'unanswered.json'
        preguntas_vistas = []
        if UNANSWERED_PATH.exists():
            with open(UNANSWERED_PATH, 'r', encoding='utf-8') as f:
                preguntas_vistas = json.load(f)
        
        if texto not in preguntas_vistas and len(texto.strip()) > 3:
            preguntas_vistas.append(texto)
            with open(UNANSWERED_PATH, 'w', encoding='utf-8') as f:
                json.dump(preguntas_vistas, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error guardando pregunta huérfana: {e}")

    return {
        'intent':    'desconocido',
        'nivel':     'fallback',
        'confianza': confianza,
        'respuesta': (
            "Lo siento, no tengo información sobre eso. "
            "Puedes llamarnos al 925 23 34 54 o escribirnos a "
            "almacen@olivillatres.com y te ayudamos."
        )
    }


# ── Rutas de Flask ───────────────────────────────────────────────────────────

@app.route('/')
def home():
    """Sirve la interfaz del chat."""
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    """Recibe los mensajes de la web."""
    data = request.get_json() or {}
    mensaje_usuario = data.get('mensaje', '')
    if not mensaje_usuario:
        return jsonify({'respuesta': 'No he recibido ningún texto.'}), 400
    
    respuesta_bot = obtener_respuesta(mensaje_usuario)
    return jsonify(respuesta_bot)


# ── HTML inline completo ───────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chatbot Empresa</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5; display: flex;
         justify-content: center; align-items: center; min-height: 100vh; }
  .chat-container { width: 420px; background: white; border-radius: 16px;
                    box-shadow: 0 4px 24px rgba(0,0,0,0.12); display: flex;
                    flex-direction: column; height: 620px; }
  .chat-header { background: #4f46e5; color: white; padding: 18px 20px;
                 border-radius: 16px 16px 0 0; display: flex; align-items: center; gap: 12px; }
  .avatar { width: 36px; height: 36px; background: rgba(255,255,255,0.25);
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: 18px; }
  .chat-header h1 { font-size: 16px; font-weight: 600; }
  .chat-header p  { font-size: 12px; opacity: 0.8; }
  .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex;
              flex-direction: column; gap: 12px; }
  .msg { display: flex; gap: 8px; max-width: 85%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .bubble { padding: 10px 14px; border-radius: 16px; font-size: 14px; line-height: 1.5; }
  .msg.bot  .bubble { background: #f1f0ff; color: #1e1b4b; border-radius: 4px 16px 16px 16px; }
  .msg.user .bubble { background: #4f46e5; color: white; border-radius: 16px 4px 16px 16px; }
  .meta { font-size: 10px; opacity: 0.55; margin-top: 3px; display: block;}
  .input-area { padding: 14px 16px; border-top: 1px solid #e5e7eb; display: flex; gap: 8px; }
  input[type=text] { flex: 1; padding: 10px 14px; border: 1.5px solid #e5e7eb;
                     border-radius: 24px; outline: none; font-size: 14px; }
  input[type=text]:focus { border-color: #4f46e5; }
  button { background: #4f46e5; color: white; border: none; padding: 10px 18px;
           border-radius: 24px; cursor: pointer; font-size: 14px; font-weight: 500; }
  button:hover { background: #4338ca; }
  .suggestions { padding: 0 16px 14px; display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: #f1f0ff; color: #4f46e5; border: 1px solid #c7d2fe;
          border-radius: 20px; padding: 4px 12px; font-size: 12px; cursor: pointer; }
  .chip:hover { background: #e0e7ff; }
</style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">
    <div class="avatar">🤖</div>
    <div><h1>Asistente Virtual</h1><p>Siempre disponible</p></div>
  </div>
  <div class="messages" id="messages">
    <div class="msg bot">
      <div>
        <div class="bubble">¡Hola! Soy el asistente virtual de Olivilla Tres. ¿En qué puedo ayudarte?</div>
      </div>
    </div>
  </div>
  <div class="suggestions">
    <span class="chip" onclick="enviarChip(this)">¿Qué horario tenéis?</span>
    <span class="chip" onclick="enviarChip(this)">Tengo goteras en la terraza</span>
    <span class="chip" onclick="enviarChip(this)">Métodos de pago</span>
  </div>
  <div class="input-area">
    <input type="text" id="user-input" placeholder="Escribe tu pregunta..."
           onkeydown="if(event.key==='Enter') enviar()">
    <button onclick="enviar()">Enviar</button>
  </div>
</div>

<script>
  function agregarMensaje(texto, remitente, nivel = '') {
    const contenedor = document.getElementById('messages');
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('msg', remitente);
    
    let metaHTML = '';
    if (nivel) {
      metaHTML = `<span class="meta">Origen: ${nivel}</span>`;
    }

    msgDiv.innerHTML = `<div><div class="bubble">${texto}</div>${metaHTML}</div>`;
    contenedor.appendChild(msgDiv);
    contenedor.scrollTop = contenedor.scrollHeight;
  }

  function enviarChip(elemento) {
    document.getElementById('user-input').value = elemento.innerText;
    enviar();
  }

  async function enviar() {
    const input = document.getElementById('user-input');
    const texto = input.value.trim();
    if (!texto) return;
    
    input.value = '';
    agregarMensaje(texto, 'user');

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ mensaje: texto })
      });
      const data = await res.json();
      agregarMensaje(data.respuesta, 'bot', data.nivel);
    } catch (error) {
      agregarMensaje("Error al conectar con el servidor.", "bot");
    }
  }
</script>
</body>
</html>
"""
