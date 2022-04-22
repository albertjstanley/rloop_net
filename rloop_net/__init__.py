# Keep seed consistent for developing
numpy_seed = 0
tensorflow_seed = 0
print(f"numpy seed {numpy_seed}, tensorflow seed {tensorflow_seed}")

from numpy.random import seed
seed(numpy_seed)

import tensorflow
tensorflow.random.set_seed(tensorflow_seed)