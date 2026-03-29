import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# CONFIG PAGE
# ==============================
st.set_page_config(
    page_title="Fenêtre Thérapeutique GBM",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# STYLE CSS
# ==============================
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    color:#4CAF50;
}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#1e1e1e;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    margin-bottom:20px;
}
.metric {
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TITRE
# ==============================
st.markdown('<p class="big-title">🧠 Outil IA - Fenêtre Thérapeutique GBM</p>', unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data_with_predictions.csv")

# sécuriser noms
df["drug"] = df["drug"].str.lower()

# ==============================
# SCORES
# ==============================
drug_scores = df.set_index("drug")["prediction"]
ranking = drug_scores.sort_values(ascending=False)

# ==============================
# 🔍 TEST MEDICAMENT
# ==============================
st.subheader("🔍 Tester un médicament")

drug_input = st.text_input("Entrer le nom du médicament")

if drug_input:
    drug_input_clean = drug_input.strip().lower()

    if drug_input_clean in drug_scores.index:

        score = drug_scores[drug_input_clean]

        st.markdown(f"""
        <div class="card">
            <p class="metric">💊 Médicament : {drug_input}</p>
            <p>📊 Score : <b>{score:.4f}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # 🔥 INTERPRÉTATION
        st.markdown(f"### 🧪 Score thérapeutique : {score:.4f}")

        if score > 0.05:
            st.success("🔥 Médicament efficace et peu toxique")
        elif score > 0:
            st.info("👍 Médicament légèrement efficace mais à surveiller")
        elif score > -0.05:
            st.warning("⚠️ Médicament peu efficace")
        else:
            st.error("❌ Médicament trop toxique ou inefficace")

    else:
        st.error("❌ Médicament non trouvé")
# ==============================
# APERÇU
# ==============================
st.subheader("📊 Aperçu des données")

n_rows = st.slider("Nombre de lignes à afficher", 5, 50, 10)
st.dataframe(df.head(n_rows))

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ Options")
top_n = st.sidebar.slider("Top N médicaments", 5, 50, 10)

# ==============================
# TOP DRUGS
# ==============================
top_drugs = ranking.head(top_n)

# ==============================
# LAYOUT (FIX FINAL)
# ==============================
col1, col2, col3 = st.columns([1, 1.2, 0.1])

# ---- COLONNE 1 ----
with col1:
    st.subheader("🏆 Top médicaments (IA Deep Learning)")
    st.dataframe(top_drugs)

# ---- COLONNE 2 ----
with col2:
    st.subheader("📈 Fenêtre thérapeutique (IA)")

    # récupérer données
    gbm = df.set_index("drug")["efficacy_gbm"]
    tox = df.set_index("drug")["efficacy_tox"]

    common = gbm.index.intersection(tox.index)
    gbm = gbm[common]
    tox = tox[common]

    # ✅ CORRECTION SCIENTIFIQUE
    therapeutic_score = gbm - tox

    # ✅ FIGURE PETITE
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(tox, therapeutic_score)
    ax.axhline(0, linestyle='--')

    ax.set_xlabel("Toxicité (non-GBM)")
    ax.set_ylabel("Score thérapeutique")
    ax.set_title("Fenêtre thérapeutique")

    # ✅ IMPORTANT
    st.pyplot(fig, use_container_width=False)