import streamlit as st
import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the pre-trained VGG16 model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load the mapping and tokenizer
with open('Working/mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

with open('Working/all_captions.pkl', 'rb') as f:
    all_captions = pickle.load(f)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1

# Constants
max_length = 34  # Set according to your model's max sequence length

# Load the trained caption generation model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
caption_model.load_weights('Working/image_caption_model.h5')  # Load your trained model weights

# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.expand_dims(image[0], axis=0), sequence], verbose=0)[0]
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text


# Streamlit app function
def app():
    st.title("Image Caption Generator")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = Image.open(uploaded_file)
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = preprocess_input(np.expand_dims(image_array, axis=0))

        # Generate caption
        image_features = vgg_model.predict(image_array)[0]
        caption = predict_caption(caption_model, [image_features, np.zeros((1, max_length))], tokenizer, max_length)

        # Display the image
        # Display the image
        normalized_image = image_array[0].astype('float32') / 255.0
        # Display the image with clamp parameter
        st.image(normalized_image, caption="Uploaded Image", use_column_width=True, clamp=True)

        # Display the generated caption
        st.subheader("Generated Caption:")
        st.write(caption)

# Run the app
if __name__ == '__main__':
    app()
