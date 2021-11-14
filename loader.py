# Import the needed libraries
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

# Test the Keras version
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Load wine file
wine = pd.read_csv(r'C:\Users\Riccardo\Documents\Universita\PoLiMi\Lezioni\IV Anno\ANN\riccamper\winequality.csv', sep=';')
print(wine.info())
wine.head()