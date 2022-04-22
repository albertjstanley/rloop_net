import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
from tensorflow.keras import Model
from rloop_net.config import sequence_width,single_stranded_output

# BPNET based model for Rloop prediction

def get_rloop_model():
    num_tracks = 4
    first_kernel_size = 25
    num_filters = 64
    activation = "relu"
    num_strands = 1 if single_stranded_output else 2


    input = keras.Input(shape=(sequence_width, num_tracks,))
    x = kl.Conv1D(filters=num_filters, 
                            kernel_size=first_kernel_size,
                            padding="same",
                            activation=activation)(input)

    for i in range(1,10):
        conv_x = kl.Conv1D(filters=num_filters, 
                                kernel_size=3,
                                padding="same",
                                # dilation_rate=2**i,
                                activation=activation)(x)
        x = kl.add([x,conv_x])

    bottleneck = x

    # Profile head
    profile_x = kl.Reshape((-1,1,64))(bottleneck)
    profile_x = kl.Conv2DTranspose(filters=num_strands,
                                        kernel_size=(25,1),
                                        padding="same")(profile_x)
    profile = kl.Reshape((-1,num_strands),name="profile")(profile_x)

    # Counts Head
    counts_x = kl.GlobalAveragePooling1D(data_format="channels_last")(bottleneck)
    counts = kl.Dense(num_strands,name="counts")(counts_x)

    outputs = [profile,counts]

    model = Model(inputs=input,outputs=outputs)
    return model