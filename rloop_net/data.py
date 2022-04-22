import numpy as np

from rloop_net.config import single_stranded_output, sequence_width
from tensorflow.keras.models import Sequential


batch_size = 16
def generate_dummy_dna_sequence():
  base = np.array([1,0,0,0])
  one_hot_encoded_seq = np.array([base] * sequence_width)
  return one_hot_encoded_seq # Shape: (10000,4)

def generate_example_batch_input():
  return np.repeat([generate_dummy_dna_sequence()], batch_size,axis=0)

def generate_example_profile_output():
  if single_stranded_output:
    return np.repeat([[3]],[sequence_width],axis=0)
  else:
    return np.repeat([[6,9]],[sequence_width],axis=0)

def generate_example_counts_output():
  if single_stranded_output:
    return np.array([100])
  else:
    return np.array([100,200])

def generate_example_batch_output():
  example_profile = generate_example_profile_output()
  example_counts = generate_example_counts_output()
  profile_batch = np.repeat([example_profile], batch_size,axis=0)
  counts_batch = np.repeat([example_counts], batch_size,axis=0)
  return [profile_batch, counts_batch]

class DataGenerator(Sequential):
  def __init__(self):
    pass