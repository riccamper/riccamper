# Import the needed libraries
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Test the Keras version
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)