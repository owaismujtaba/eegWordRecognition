from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,  StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pdb
import numpy as np
from src.utils import laodPCADataset

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.utils.class_weight import compute_class_weight





def CNNLSTMModel(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.RepeatVector(input_shape[0] // 4 * input_shape[1] // 4),
        #tf.keras.layers.LSTM(64, return_sequences=False),
        #tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_sequential_model(train_labels):

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = dict(enumerate(class_weights))
    print(len(np.unique(train_labels)))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(165, 1)),  # Input shape (samples, points, channels)
        tf.keras.layers.Conv1D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(256, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        #tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(512, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        #tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(1024, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation='sigmoid')  # Binary classification, so output is a single neuron with sigmoid activation
    ])

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
        )

    return model, class_weights_dict





    X, y = laodPCADataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=20),
        "SVM": SVC()
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"{name} Accuracy Test: {accuracy:.2f}")
        y_pred = model.predict(X_train_scaled)
        accuracy = accuracy_score(y_train_encoded, y_pred)
        print(f"{name} Accuracy Train: {accuracy:.2f}")