from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf
from tensorflow.keras import layers

# Load dataset.
dftrain = pd.read_csv('diabetes.csv')  # training data
dfeval = pd.read_csv('diabetes.csv')  # testing data
y_train = dftrain.pop('Outcome')
y_eval = dfeval.pop('Outcome')

NUMERIC_COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Create preprocessing layers for the numeric features
preprocessing_layers = {col: layers.Input(shape=(1,), name=col) for col in NUMERIC_COLUMNS}

print(preprocessing_layers)


# Combine the preprocessing layers
all_inputs = list(preprocessing_layers.values())
all_features = layers.concatenate([layers.Rescaling(1./np.max(dftrain[col]))(preprocessing_layers[col]) for col in NUMERIC_COLUMNS])

# Define the model
x = layers.Dense(128, activation='relu')(all_features)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=all_inputs, outputs=output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define input functions
def df_to_dataset(data_df, label_df, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_df))
    ds = ds.batch(batch_size)
    return ds

train_dataset = df_to_dataset(dftrain, y_train)
eval_dataset = df_to_dataset(dfeval, y_eval, shuffle=False)

# Train the model
model.fit(train_dataset, epochs=250)

# Evaluate the model
loss, accuracy = model.evaluate(eval_dataset)
clear_output()
print(f'Loss: {loss}, Accuracy: {accuracy}')


# Cross-check predictions vs actual values
predictions = model.predict(eval_dataset)
predicted_classes = (predictions > 0.5).astype(int).flatten()

for i in range(10):  # Display first 10 results for cross-checking
    print(f'Predicted: {predicted_classes[i]}, Actual: {y_eval.iloc[i]}')
