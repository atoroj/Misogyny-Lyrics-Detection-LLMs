import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import csv

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../../models/Model_DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit-CoT_GPT4o-R_16-Alpha_16-LR_2e-05-Tarea_1",   # Modelo base
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()  # <- sí se puede usar

FLAG_FILE = "flags_data/flags.csv"
os.makedirs(os.path.dirname(FLAG_FILE), exist_ok=True)

def clean_lyrics(text):
    # Elimina caracteres no alfabéticos (excepto espacios y letras acentuadas comunes en español)
    text = re.sub(r"[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ ]+", " ", text)
    # Convierte a minúsculas
    text = text.lower()
    # Reduce espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Función de predicción
def detect_misogyny(text):
    cleaned_text = clean_lyrics(text)
    # Construir el prompt de entrada
    prompt = """
            ### Instruccion
            Analiza la siguiente letra de canción y determina si contiene contenido misógino. Evalúa si incluye lenguaje, actitudes o mensajes que:
              - Degraden o deshumanicen a las mujeres.
              - Menosprecien a las mujeres de manera explícita o implícita.
              - Refuercen estereotipos negativos o dañinos sobre las mujeres.
              - Promuevan violencia física, emocional o sexual contra las mujeres.
            Piensa cuidadosamente tu respuesta y crea paso a paso una chain of thoughts para dar una respuesta logica.
            Responde únicamente con "1" si la letra es misógina o con "0" si la letra no es misógina. No proporciones ninguna explicación ni texto adicional.

            ### Letra:
            {lyrics}

            ### Respuesta:
            <think>"""
    
    prompt = prompt.format(lyrics=text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.6
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer explicación entre <think>...</think>
    explanation_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    # Extraer "1" o "0" después de </think>
    label_match = re.search(r"</think>\s*(\d)", response)
    label = label_match.group(1) if label_match else ""

    # Combinar resultado final
    return f"{explanation}\n\nRespuesta final: {label}" if explanation and label else response.strip()


def save_flag(user_text, response, flag_type):
    # Guarda la entrada, salida y si fue correcta o incorrecta en CSV
    with open(FLAG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_text, response, flag_type])
    return f"Guardado flag: {flag_type}"

with gr.Blocks() as demo:
    user_input = gr.Textbox(label="Letra de canción", lines=10)
    result = gr.Textbox(label="Respuesta del modelo", lines=10)
    
    btn_analizar = gr.Button("Analizar")
    btn_correcto = gr.Button("Respuesta correcta")
    btn_incorrecto = gr.Button("Respuesta incorrecta")
    
    btn_analizar.click(fn=detect_misogyny, inputs=user_input, outputs=result)
    
    btn_correcto.click(fn=save_flag, inputs=[user_input, result, gr.State("correcto")], outputs=result)
    btn_incorrecto.click(fn=save_flag, inputs=[user_input, result, gr.State("incorrecto")], outputs=result)

demo.launch()