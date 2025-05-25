# API de Predicción de Géneros de Películas

Este proyecto implementa una API REST que predice géneros de películas basándose en la sinopsis. Utiliza un modelo de Regresión Logística entrenado con datos de sinopsis de películas y sus géneros correspondientes.

## Características

- Predicción de probabilidades para 24 géneros de películas diferentes
- Preprocesamiento automático de texto para mejorar las predicciones
- API REST fácil de usar con Flask y Flask-RESTX
- Documentación interactiva con Swagger UI

## Requisitos

- Python 3.8+
- Flask
- Flask-RESTX
- NLTK
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/MIAD_2025_ML_P2.git
cd MIAD_2025_ML_P2
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. Descargar los recursos de NLTK (se hace automáticamente al iniciar la API)

## Uso

1. Iniciar el servidor API:
```bash
python api.py
```

2. Acceder a la documentación Swagger UI:
```
http://localhost:5000/
```

3. Realizar una predicción mediante una solicitud GET:
```
http://localhost:5000/predict/?plot=A young man discovers he has special powers and embarks on a journey to save the world from an ancient evil.
```

## Estructura del Proyecto

- `api.py`: Código principal de la API
- `mejor_modelo_peliculas.pkl`: Modelo entrenado de Regresión Logística
- `vectorizador_peliculas.pkl`: Vectorizador TF-IDF para procesar el texto
- `mlb_peliculas.pkl`: MultiLabelBinarizer para codificar los géneros

## Modelo

El modelo utilizado es una Regresión Logística con C=1.0, que obtuvo un AUC de 0.9267 en validación cruzada. El modelo fue entrenado con sinopsis de películas y puede predecir 24 géneros diferentes.

## Licencia

Este proyecto está bajo la licencia MIT.
