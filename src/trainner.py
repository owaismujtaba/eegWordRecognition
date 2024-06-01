from src.models import CNNLSTMModel

import config

import os

from src.data_loader import getDataLoaders

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import pandas as pd
import pdb

import tensorflow as tf

# Example of reducing batch size
batch_size = 16  # Reduce from 32 to 16

# Example of using mixed precision
from tensorflow.keras import mixed_precision




def train():
    
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    train_ds, val_ds = getDataLoaders()
    
    print('loading model')
    model = CNNLSTMModel((config.imageSize, config.imageSize, 3), config.nClasses)

    
    
    loss = SparseCategoricalCrossentropy()

    model.compile(
        optimizer=Adam(),
        loss=loss,
        metrics=['accuracy']
    )

    print(model.summary())
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    print('training started')
    history = model.fit(train_ds, epochs=config.epochs, callbacks=[early_stopping], validation_data=val_ds)
    history = pd.DataFrame(history.history)
    history.save('history.csv')