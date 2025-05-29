# Misogyny Detection in Song Lyrics using Large Language Models  
Antonio Toro Jaén – Trabajo de Fin de Grado – Grado en Ingeniería Informática

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/huggingface-FFD14E?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace">
  <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/google%20colab-F9AB00?style=for-the-badge&logo=google%20colab&logoColor=white" alt="Google Colab">
  <img src="https://img.shields.io/badge/GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="GPU">
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  <img src="https://img.shields.io/badge/latex-008080?style=for-the-badge&logo=latex&logoColor=white" alt="LaTeX">
</p>

Este repositorio contiene el trabajo realizado para el Trabajo de Fin de Grado **"Detección y Clasificación de Discursos Misóginos en Letras de Canciones con Deep Learning"** presentado en la competición MiSonGyny 2025 (IberLEF).

<p align="center">
  <img src="https://www.hazfundacion.org/wp-content/uploads/2022/11/universidad-de-huelva.png" alt="Universidad de Huelva" width="600">
</p>

## Contenido del Repositorio

### Estructura de Directorios

- **Dataset/**: Conjuntos de datos de entrenamiento y test utilizados en ambas tareas.
- **Cuadernos/**: Cuadernos de entrenamiento y evaluación de modelos (DeepSeek-R1, LLaMA 3.1, Qwen3).
- **Resultados/**: Archivos de predicción y resultados finales enviados a la competición.
- **App/**: Código de la aplicación desarrollada con Gradio.

### Cuadernos Utilizados


1. **mysongyny_CON_CoT_Tarea1.ipynb**:  Fine-tuning con DeepSeek-R1 usando razonamiento (Chain of Thought) para Tarea 1.
2. **mysongyny_CON_CoT_Tarea2.ipynb**: Fine-tuning con DeepSeek-R1 usando razonamiento (Chain of Thought) para Tarea 2.
3. **mysongyny_SIN_CoT_Tarea1**: Mejores resultados en Task 1. Prompt directo.
4. **mysongyny_SIN_CoT_Tarea2.ipynb**: Mejores resultados en Task 2. Prompt directo.
3. **longformerbaseline**:  Primer baseline desarrollado, basado en un modelo Longformer, usado como referencia.

### Repositorios en HuggingFace
1. [**DeepSeekR1-MiSonGyny para Tarea 1**](https://huggingface.co/atorojaen/DeepSeek-R1-MiSonGyny)
2. [**Llama3.1-MiSonGyny para Tarea 1**](https://huggingface.co/atorojaen/Llama3.1-MiSonGyny)
3. [**Qwen3-MiSonGyny para Tarea 1**](https://huggingface.co/atorojaen/Qwen3-MiSonGyny)
4. [**Qwen3-MiSonGyny para Tarea 2**](https://huggingface.co/atorojaen/Qwen3-MiSonGyny-Task2)

### Resultados

Los resultados obtenidos se encuentran en la carpeta `Resultados/`. El sistema propuesto obtuvo la **segunda posición** en ambas tareas de la competición.

## Tecnologías Utilizadas

- **Python**: Lenguaje principal para el procesamiento y entrenamiento de modelos.
- **Jupyter**: Cuadernos interactivos para desarrollo y experimentación.
- **HuggingFace Transformers**: Fine-tuning y tokenización de modelos LLM.
- **Unsloth**: Librería optimizada para el fine-tuning eficiente de LLMs con LoRA.
- **PyTorch**: Framework de deep learning.
- **Google Colab**: Entrenamiento en la nube con GPU.
- **GPU laboratorio (RTX 4070, 12GB VRAM)**: Entrenamiento de modelos.
- **GitHub**: Control de versiones y repositorio de código.
- **LaTeX**: Redacción del artículo científico y memoria.
