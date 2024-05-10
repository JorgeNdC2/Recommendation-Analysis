import re
import nltk
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import streamlit as st


# Funciones de preprocesamiento
def remove_between_square_brackets(text):
    """
    Remove text between square brackets (including the brackets) from the input text.

    Args:
    - text (str): The input text.

    Returns:
    - str: The text with content between square brackets removed.
    """
    if isinstance(text, str):  # Check if the text is a string
        return re.sub(r'\[[^]]*\]', '', text)
    else:
        return text

def denoise_text(text):
    """
    Perform text denoising by removing HTML tags and text between square brackets.

    Args:
    - text (str): The input text.

    Returns:
    - str: The denoised text without HTML tags and content between square brackets.
    """
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text, remove_digits=True):
    """
    Remove special characters from the input text.

    Args:
    - text (str): The input text.
    - remove_digits (bool): Whether to remove digits or not. Default is True.

    Returns:
    - str: The text with special characters removed.
    """
    if isinstance(text, str):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', text)
        if remove_digits:
            text = re.sub(r'\d+', '', text)  # Remove digits if specified
        return text
    else:
        return text

def simple_stemmer(text):
    """
    Perform simple stemming on the input text using the NLTK Porter Stemmer.

    Args:
    - text (str): The input text.

    Returns:
    - str: The text with words stemmed using the Porter Stemmer.
    """
    if isinstance(text, str):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text
    else:
        return text

def remove_stopwords(text, is_lower_case=False):
    """
    Remove stopwords from the input text.

    Args:
    - text (str): The input text.
    - is_lower_case (bool): Whether the text is lowercased or not. Default is False.

    Returns:
    - str: The text with stopwords removed.
    """
    nltk.download('punkt')
    nltk.download('stopwords')

    # Tokenizador y lista de stopwords
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Función de procesamiento de la frase completa
def preprocess_sentence(sentence):
    """
    Preprocess a sentence by applying denoising, removing special characters,
    stemming, and removing stopwords.

    Args:
    - sentence (str): The input sentence.

    Returns:
    - str: The preprocessed sentence.
    """
    # Remover texto entre corchetes
    sentence = denoise_text(sentence)
    # Remover caracteres especiales
    sentence = remove_special_characters(sentence)
    # Realizar stemming
    sentence = simple_stemmer(sentence)
    # Remover stopwords
    sentence = remove_stopwords(sentence)
    return sentence




def procesar_texto(texto):
    # Cargar el modelo
    modelo_cargado = load_model("modelo_sentimientos.h5")

    # Tokenize the texts
    max_words = 8000  # Consider only the top 8,000 words
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([texto])  # Asegúrate de que sea una lista de textos
    sequences = tokenizer.texts_to_sequences([texto])  # Asegúrate de que sea una lista de textos

    # Pad sequences
    maxlen = 500  # Cut texts after this number of words
    data = pad_sequences(sequences, maxlen=maxlen)

    print(data)
    print(data.shape)
    print(type(data))

    prediccion = modelo_cargado.predict(data)
    print(prediccion)

    # Interpretar la predicción
    if prediccion[0] >= 0.35:
        return True
    else:
        return False


def main():
    st.title("Aplicación de Procesamiento de Texto")

    # Recomendaciones y descripción
    st.write("""
    Bienvenido a la Aplicación de Procesamiento de Texto. Esta herramienta te ayudará a determinar si una prenda de ropa es adecuada o no.
    Introduce una cadena de texto en el cuadro de abajo y presiona el botón 'Procesar' para obtener una recomendación.
    Nota: Esto solo es para interectuar con el modelo y jugar un poco, debido a la particularidad de los datos de entrenamiento y su pequeño tamaño, la aplicación overfitea mucho,
    además recordar que el texto tiene que ser en inglés, y cuanto más largo mejor. 
    Dicho esto, prueba lo que quieras!! :).
    """)

    texto_usuario = st.text_input("Introduce una cadena de texto:")

    if st.button("Procesar"):
        resultado_procesamiento = procesar_texto(texto_usuario)

        # Mostrar resultado
        st.subheader("Resultado del Procesamiento:")

        if resultado_procesamiento:
            st.write("¡La prenda es recomendada!")
            st.markdown(
                "**Resultado:** <span style='color:green'>RECOMENDADA</span>",
                unsafe_allow_html=True)
        else:
            st.write("¡La prenda no es recomendada!")
            st.markdown(
                "**Resultado:** <span style='color:red'>NO RECOMENDADA</span>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
