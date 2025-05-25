# Importación librerías
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd
import os
import re
import string
import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='API to predict movie genres based on plot synopsis using Logistic Regression model')

ns = api.namespace('predict', 
     description='Movie Genre Classifier')

# Definición argumentos o parámetros de la API
parser = api.parser()

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Plot synopsis of the movie', 
    location='args')

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Open Multilingual WordNet

# Definir posibles rutas para los archivos de modelo
posible_paths = [
    # Directorio actual
    os.path.join(os.getcwd(), 'vectorizador_peliculas.pkl'),
    os.path.join(os.getcwd(), 'mlb_peliculas.pkl'),
    os.path.join(os.getcwd(), 'mejor_modelo_peliculas.pkl'),
    
    # Directorio padre
    os.path.join(os.path.dirname(os.getcwd()), 'vectorizador_peliculas.pkl'),
    os.path.join(os.path.dirname(os.getcwd()), 'mlb_peliculas.pkl'),
    os.path.join(os.path.dirname(os.getcwd()), 'mejor_modelo_peliculas.pkl'),
    
    # Directorio del script
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorizador_peliculas.pkl'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlb_peliculas.pkl'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mejor_modelo_peliculas.pkl'),
    
    # Directorio padre del script
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vectorizador_peliculas.pkl'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlb_peliculas.pkl'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mejor_modelo_peliculas.pkl'),
]

# Imprimir rutas para depuración
print("Buscando modelos en las siguientes rutas:")
for path in posible_paths[:3]:  # Solo mostrar las primeras 3 rutas para no saturar la salida
    print(f"- {path}")
print("...")

# Cargar modelos
vectorizer_path = None
mlb_path = None
model_path = None

# Buscar archivos existentes
for path in posible_paths:
    if os.path.exists(path):
        if 'vectorizador_peliculas.pkl' in path:
            vectorizer_path = path
        elif 'mlb_peliculas.pkl' in path:
            mlb_path = path
        elif 'mejor_modelo_peliculas.pkl' in path:
            model_path = path

# Verificar si se encontraron todos los archivos
if not (vectorizer_path and mlb_path and model_path):
    missing = []
    if not vectorizer_path:
        missing.append('vectorizador_peliculas.pkl')
    if not mlb_path:
        missing.append('mlb_peliculas.pkl')
    if not model_path:
        missing.append('mejor_modelo_peliculas.pkl')
    
    error_msg = f"No se pudieron encontrar los siguientes archivos: {', '.join(missing)}"
    print(error_msg)
    sys.exit(1)

# Cargar los modelos
try:
    print(f"Cargando vectorizador desde: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    
    print(f"Cargando mlb desde: {mlb_path}")
    mlb = joblib.load(mlb_path)
    
    print(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    
    print("Todos los modelos cargados correctamente")
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    sys.exit(1)

# Funciones de preprocesamiento
def obtener_stopwords_extendidas():
    """
    Obtiene una lista extendida de stopwords, incluyendo palabras comunes
    que aparecen en todos los géneros y no son discriminativas.
    """
    # Stopwords básicas de NLTK
    stop_words = set(stopwords.words('english'))
    
    # Añadir palabras adicionales que son comunes en sinopsis pero no discriminativas
    palabras_adicionales = {
        'film', 'movie', 'story', 'man', 'woman', 'find', 'way', 'day', 'time', 'year',
        'new', 'old', 'young', 'life', 'live', 'take', 'make', 'made', 'get', 'got',
        'come', 'came', 'go', 'went', 'gone', 'see', 'saw', 'seen', 'know', 'knew',
        'known', 'want', 'wanted', 'need', 'needed', 'tell', 'told', 'say', 'said',
        'ask', 'asked', 'try', 'tried', 'help', 'helped', 'call', 'called', 'feel',
        'felt', 'become', 'became', 'leave', 'left', 'put', 'set', 'end', 'start',
        'began', 'begin', 'begun', 'show', 'shown', 'give', 'gave', 'given'
    }
    
    # Unir stopwords
    stop_words.update(palabras_adicionales)
    
    return stop_words

def limpiar_texto(texto):
    """
    Limpieza básica de texto.
    """
    if not isinstance(texto, str):
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar puntuación
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    
    # Eliminar números
    texto = re.sub(r'\d+', '', texto)
    
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def preprocesar_texto(texto):
    """
    Preprocesamiento completo de texto.
    """
    # Obtener stopwords
    stop_words = obtener_stopwords_extendidas()
    
    # Limpiar texto
    texto_limpio = limpiar_texto(texto)
    
    # Tokenizar
    tokens = word_tokenize(texto_limpio)
    
    # Eliminar stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lematizar
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Unir tokens
    texto_final = ' '.join(tokens)
    
    return texto_final

# Definir modelo de respuesta para la API
# Definir los campos para cada género específico
resource_fields = api.model('Resource', {
    'p_Action': fields.Float,
    'p_Adventure': fields.Float,
    'p_Animation': fields.Float,
    'p_Biography': fields.Float,
    'p_Comedy': fields.Float,
    'p_Crime': fields.Float,
    'p_Documentary': fields.Float,
    'p_Drama': fields.Float,
    'p_Family': fields.Float,
    'p_Fantasy': fields.Float,
    'p_Film-Noir': fields.Float,
    'p_History': fields.Float,
    'p_Horror': fields.Float,
    'p_Music': fields.Float,
    'p_Musical': fields.Float,
    'p_Mystery': fields.Float,
    'p_News': fields.Float,
    'p_Romance': fields.Float,
    'p_Sci-Fi': fields.Float,
    'p_Short': fields.Float,
    'p_Sport': fields.Float,
    'p_Thriller': fields.Float,
    'p_War': fields.Float,
    'p_Western': fields.Float
})

# Definición de la clase para disponibilización
@ns.route('/')
class MovieGenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        # Obtener la sinopsis
        plot = args['plot']
        
        # Preprocesar la sinopsis
        plot_processed = preprocesar_texto(plot)
        
        # Vectorizar la sinopsis
        plot_vectorized = vectorizer.transform([plot_processed])
        
        # Obtener probabilidades para cada género
        genre_probabilities = model.predict_proba(plot_vectorized)[0]
        
        # Crear diccionario con las probabilidades para cada género
        result = {}
        
        # Lista de todos los géneros posibles
        all_genres = [
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
            'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
            'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western'
        ]
        
        # Inicializar todas las probabilidades a 0
        for genre in all_genres:
            result[f'p_{genre}'] = 0.0
        
        # Asignar las probabilidades reales para los géneros presentes en el modelo
        for i, genre in enumerate(mlb.classes_):
            result[f'p_{genre}'] = float(genre_probabilities[i])
        
        return result, 200

if __name__ == '__main__':
    # Configuración para entorno de producción
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando servidor API en http://localhost:{port}")
    print("Para probar el API, usa la siguiente URL (reemplaza 'sinopsis_ejemplo' con tu texto):")
    print(f"http://localhost:{port}/predict/?plot=sinopsis_ejemplo")
    app.run(debug=True, host='0.0.0.0', port=port)
