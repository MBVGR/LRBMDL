import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

LANGUAGES = ["Telugu", "Hindi", "Tamil", "Kannada", "English"]

def build_model():
    inputs = layers.Input(shape=(50, 200, 1))
    
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    # After MaxPooling2D((2,2)): shape = (25, 100, 32)
    # Reshape to (time_steps=100, features=25*32=800)
    x = layers.Reshape(target_shape=(100, 25 * 32))(x)  # ✅ Fixed: (100, 800)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Head 1: Language ID
    l_feat = layers.Flatten()(x)
    l_out = layers.Dense(len(LANGUAGES), activation='softmax', name='lang_out')(l_feat)

    # Head 2: Text Recognition
    t_out = layers.TimeDistributed(layers.Dense(100, activation='softmax'), name='text_out')(x)

    model = models.Model(inputs=inputs, outputs=[l_out, t_out])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/unified_model.h5')
    print("SUCCESS: models/unified_model.h5 created.")
    return model

if __name__ == "__main__":
    build_model()