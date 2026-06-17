"""
chatbot/app/app.py
==================
Servidor Flask con interfaz web para el chatbot.
"""

import pickle
import random
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import normalizar, predecir

app = Flask(__name__)

# Cargar modelo al iniciar
BASE = Path(__file__).parent
MODELO_PATH = BASE / 'chatbot_model.pkl'

with open(MODELO_PATH, 'rb') as f:
    modelo_data = pickle.load(f)

pipeline   = modelo_data['pipeline']
respuestas = modelo_data['respuestas']


def obtener_respuesta(texto: str):
    intent, confianza = predecir(pipeline, texto)
    if intent == 'desconocido' or intent not in respuestas:
        return {
            'intent': 'desconocido',
            'confianza': confianza,
            'respuesta': "Lo siento, no he entendido tu pregunta. "
                         "¿Puedes reformularla o preguntar por horarios, productos, contacto, envíos o devoluciones?"
        }
    return {
        'intent': intent,
        'confianza': round(confianza, 3),
        'respuesta': random.choice(respuestas[intent])
    }


# ── HTML inline (sin ficheros estáticos externos) ────────────────────────────

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
  .meta { font-size: 11px; opacity: 0.55; margin-top: 3px; }
  .msg.user .meta { text-align: right; }
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
        <div class="bubble">¡Hola! Soy el asistente virtual de la empresa. Puedo ayudarte con información sobre productos, horarios, envíos, devoluciones y más. ¿En qué puedo ayudarte?</div>
      </div>
    </div>
  </div>
  <div class="suggestions">
    <span class="chip" onclick="enviarChip(this)">¿Qué horario tenéis?</span>
    <span class="chip" onclick="enviarChip(this)">¿Cómo hago una devolución?</span>
    <span class="chip" onclick="enviarChip(this)">Métodos de pago</span>
    <span class="chip" onclick="enviarChip(this)">Plazos de envío</span>
  </div>
  <div class="input-area">
    <input type="text" id="user-input" placeholder="Escribe tu pregunta..." onkeydown="if(event.key==='Enter') enviar()">
    <button onclick="enviar()">Enviar</button>
  </div>
</div>
<script>
  async function enviar() {
    const input = document.getElementById('user-input');
    const texto = input.value.trim();
    if (!texto) return;
    input.value = '';
    agregarMensaje(texto, 'user');
    const res  = await fetch('/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mensaje: texto})
    });
    const data = await res.json();
    agregarMensaje(data.respuesta, 'bot', data.intent, data.confianza);
  }
  function enviarChip(el) {
    document.getElementById('user-input').value = el.textContent;
    enviar();
  }
  function agregarMensaje(texto, tipo, intent='', confianza=null) {
    const cont = document.getElementById('messages');
    const div  = document.createElement('div');
    div.className = 'msg ' + tipo;
    let meta = '';
    if (tipo === 'bot' && intent) {
      meta = `<div class="meta">intent: ${intent} · confianza: ${(confianza*100).toFixed(0)}%</div>`;
    }
    div.innerHTML = `<div><div class="bubble">${texto}</div>${meta}</div>`;
    cont.appendChild(div);
    cont.scrollTop = cont.scrollHeight;
  }
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'mensaje' not in data:
        return jsonify({'error': 'Falta el campo mensaje'}), 400
    resultado = obtener_respuesta(data['mensaje'])
    return jsonify(resultado)


@app.route('/intents', methods=['GET'])
def listar_intents():
    """Endpoint de utilidad: lista los intents disponibles."""
    return jsonify({'intents': list(respuestas.keys())})


if __name__ == '__main__':
    print("Servidor iniciado en http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)