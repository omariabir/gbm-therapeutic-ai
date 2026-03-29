# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:46:23 2026

@author: LENOVO
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# charger données
df = pd.read_csv("data.csv")

# créer efficacité
df["efficacy"] = 1 - df["response"]

# encoder
df["drug_id"] = df["drug"].astype("category").cat.codes
df["cell_id"] = df["cell_line"].astype("category").cat.codes

X_drug = df["drug_id"].values
X_cell = df["cell_id"].values
y = df["efficacy"].values

# split
X_drug_train, X_drug_test, X_cell_train, X_cell_test, y_train, y_test = train_test_split(
    X_drug, X_cell, y, test_size=0.2
)

# tailles
n_drugs = df["drug_id"].nunique()
n_cells = df["cell_id"].nunique()

# modèle
input_drug = Input(shape=(1,))
input_cell = Input(shape=(1,))

emb_drug = Embedding(n_drugs, 8)(input_drug)
emb_cell = Embedding(n_cells, 8)(input_cell)

flat_drug = Flatten()(emb_drug)
flat_cell = Flatten()(emb_cell)

x = Concatenate()([flat_drug, flat_cell])
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
output = Dense(1)(x)

model = Model(inputs=[input_drug, input_cell], outputs=output)
model.compile(optimizer="adam", loss="mse")

# entraîner
model.fit([X_drug_train, X_cell_train], y_train, epochs=3, batch_size=256)

# prédictions
df["prediction"] = model.predict([X_drug, X_cell]).flatten()

# sauvegarder
df.to_csv("data_with_predictions.csv", index=False)

print("✅ Modèle entraîné et fichier sauvegardé !")