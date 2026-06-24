"""
app.py - Versión ligera sin retriever. Solo TF-IDF + SVM + intents.json.
"""

import pickle
import random
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import sys

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from train import normalizar, predecir, entrenar
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Cargar datos ──────────────────────────────────────────────────────────────

MODELO_PATH  = BASE / 'chatbot_model.pkl'
INTENTS_PATH = BASE / 'intents.json'

with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

entrenar(ruta_datos=str(INTENTS_PATH), ruta_modelo=str(MODELO_PATH))

with open(MODELO_PATH, 'rb') as f:
    modelo_data = pickle.load(f)

pipeline   = modelo_data['pipeline']
respuestas = modelo_data['respuestas']


# ── Lógica de respuesta ───────────────────────────────────────────────────────

def obtener_respuesta(texto: str):
    texto_norm = normalizar(texto)

    # Nivel 1: coincidencia exacta
    for intent in intents_data["intents"]:
        patrones = [normalizar(p) for p in intent["patterns"]]
        if texto_norm in patrones:
            return {
                "intent":    intent["tag"],
                "nivel":     "exacto",
                "confianza": 1.0,
                "respuesta": random.choice(intent["responses"])
            }

    # Nivel 2: SVM
    intent, confianza = predecir(pipeline, texto)

    if intent != 'desconocido' and intent in respuestas:
        return {
            'intent':    intent,
            'nivel':     'svm',
            'confianza': round(confianza, 3),
            'respuesta': random.choice(respuestas[intent])
        }

    # Nivel 3: fallback + guardar pregunta
    try:
        UNANSWERED_PATH = BASE / 'unanswered.json'
        preguntas = []
        if UNANSWERED_PATH.exists():
            with open(UNANSWERED_PATH, 'r', encoding='utf-8') as f:
                preguntas = json.load(f)
        if texto not in preguntas and len(texto.strip()) > 3:
            preguntas.append(texto)
            with open(UNANSWERED_PATH, 'w', encoding='utf-8') as f:
                json.dump(preguntas, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error guardando pregunta: {e}")

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


# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    mensaje = data.get('mensaje', '')
    if not mensaje:
        return jsonify({'respuesta': 'No he recibido ningún texto.'}), 400
    return jsonify(obtener_respuesta(mensaje))


# ── HTML inline ───────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chatbot Olivilla Tres</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f4f2f8; display: flex;
         justify-content: center; align-items: center; min-height: 100vh; }
  .chat-container { width: 420px; background: white; border-radius: 16px;
                    box-shadow: 0 4px 24px rgba(74,59,104,0.15); display: flex;
                    flex-direction: column; height: 620px; }
  .chat-header { background: #4a3b68; color: white; padding: 18px 20px;
                 border-radius: 16px 16px 0 0; display: flex; align-items: center; gap: 12px;
                 border-bottom: 3px solid #b53987; }
  .avatar { width: 36px; height: 36px; background: rgba(255,255,255,0.2);
            border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; }
  .chat-header h1 { font-size: 16px; font-weight: 600; }
  .chat-header p  { font-size: 12px; opacity: 0.75; }
  .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .msg { display: flex; gap: 8px; max-width: 85%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .bubble { padding: 10px 14px; border-radius: 16px; font-size: 14px; line-height: 1.5; }
  .msg.bot  .bubble { background: #f7f0f5; color: #4a3b68; border-radius: 4px 16px 16px 16px; }
  .msg.user .bubble { background: #4a3b68; color: white; border-radius: 16px 4px 16px 16px; }
  .input-area { padding: 14px 16px; border-top: 1px solid rgba(74,59,104,0.12); display: flex; gap: 8px; }
  input[type=text] { flex: 1; padding: 10px 14px; border: 1.5px solid rgba(74,59,104,0.2);
                     border-radius: 24px; outline: none; font-size: 14px; }
  input[type=text]:focus { border-color: #b53987; }
  button { background: #b53987; color: white; border: none; padding: 10px 18px;
           border-radius: 24px; cursor: pointer; font-size: 14px; font-weight: 500; }
  button:hover { background: #922f6d; }
  .suggestions { padding: 0 16px 14px; display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: #f7f0f5; color: #4a3b68; border: 1px solid rgba(74,59,104,0.2);
          border-radius: 20px; padding: 4px 12px; font-size: 12px; cursor: pointer; }
  .chip:hover { background: #ede5f5; }
</style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">
    <div class="avatar">🤖</div>
    <div><h1>OliviBot</h1><p>Asistente de Olivilla Tres</p></div>
  </div>
  <div class="messages" id="messages">
    <div class="msg bot"><div><div class="bubble">¡Hola! Soy OliviBot, el asistente de Olivilla Tres. ¿En qué puedo ayudarte?</div></div></div>
  </div>
  <div class="suggestions">
    <span class="chip" onclick="enviarChip(this)">¿Qué horario tenéis?</span>
    <span class="chip" onclick="enviarChip(this)">Tengo goteras en la terraza</span>
    <span class="chip" onclick="enviarChip(this)">Métodos de pago</span>
    <span class="chip" onclick="enviarChip(this)">Impermeabilizantes</span>
  </div>
  <div class="input-area">
    <input type="text" id="user-input" placeholder="Escribe tu pregunta..."
           onkeydown="if(event.key==='Enter') enviar()">
    <button onclick="enviar()">Enviar</button>
  </div>
</div>
<script>
  function agregarMensaje(texto, remitente) {
    const c = document.getElementById('messages');
    const d = document.createElement('div');
    d.classList.add('msg', remitente);
    d.innerHTML = `<div><div class="bubble">${texto}</div></div>`;
    c.appendChild(d);
    c.scrollTop = c.scrollHeight;
  }
  function enviarChip(el) { document.getElementById('user-input').value = el.innerText; enviar(); }
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
      agregarMensaje(data.respuesta, 'bot');
    } catch { agregarMensaje('Error al conectar con el servidor.', 'bot'); }
  }
</script>
</body>
</html>"""