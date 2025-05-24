# Custom L1 Distance layer module

# Import dependencies 
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Layer
class L1Dist(Layer):

    # Init method / Inicijalizacija 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Similarity method / Metoda sličnosti  
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)