import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

# Paths
DATASET_PATH = "dataset/Flickr8k"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
CAPTIONS_FILE = os.path.join(DATASET_PATH, "captions.txt")

# Set hyperparameters
VOCAB_SIZE = 5000
MAX_LENGTH = 35
EMBEDDING_DIM = 256
BATCH_SIZE = 32
EPOCHS = 10

def extract_features(directory):
    model = ResNet50()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        features[img_name] = feature
    return features

def data_generator(descriptions, features, tokenizer, max_length, vocab_size, n_step):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            n += 1
            feature = features[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == n_step:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0

def build_model(vocab_size, max_length):
    # CNN Model features
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM Sequence generator
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

if __name__ == "__main__":
    print("Starting Training Pipeline...")
    # 1. Feature Extraction (Dummy call for now)
    # print("Extracting features (This may take a while)...")
    # features = extract_features(IMAGES_PATH)
    # pickle.dump(features, open("model/features.pkl", "wb"))

    # 2. Build and Train
    # model = build_model(VOCAB_SIZE, MAX_LENGTH)
    # model.fit(...)
    print("Model Architecture Created. Ready for data.")
    print("Save LSTM model to model/lstm_model.h5")
