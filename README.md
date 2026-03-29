# 🧠 IA pour la Fenêtre Thérapeutique du Glioblastome (GBM)

## 🎯 Objectif du projet

Ce projet utilise le deep learning pour identifier des médicaments efficaces contre le glioblastome (GBM), tout en minimisant leur toxicité sur les cellules normales.

👉 L'objectif est de trouver des molécules qui :
- détruisent les cellules tumorales (GBM)
- n'endommagent pas les cellules saines

---

## 🧪 Données

Les données contiennent :
- médicaments (`drug`)
- lignées cellulaires (`cell_line`)
- réponse biologique (`response`)
- type de cellule (`is_gbm`)

---

## 🤖 Modèle utilisé

Le modèle est un réseau de neurones avec embeddings :

- 🔹 Embedding des médicaments
- 🔹 Embedding des cellules
- 🔹 Couches Dense (Deep Learning)

Architecture :
- Embedding → Flatten → Concatenate
- Dense (64) → Dense (32) → Output

---

## 📊 Score thérapeutique

On définit un score scientifique :

Score thérapeutique = efficacité (GBM) - toxicité (non-GBM)

👉 Interprétation :
- Score > 0 → bon candidat
- Score < 0 → trop toxique ou inefficace

---

## 📈 Visualisation

L'application affiche une **fenêtre thérapeutique** :

- Axe X → toxicité (cellules normales)
- Axe Y → score thérapeutique (GBM - non-GBM)

👉 Un bon médicament est :
- en haut (efficace)
- au-dessus de 0 (sélectif)
- pas trop à gauche (pas toxique)

---

## 💻 Application (Streamlit)

L'application permet :

- 🔍 Tester un médicament
- 🏆 Voir les meilleurs candidats
- 📊 Visualiser la fenêtre thérapeutique

---

## 🚀 Lancer le projet

Installer les dépendances :

```bash
pip install -r requirements.txt