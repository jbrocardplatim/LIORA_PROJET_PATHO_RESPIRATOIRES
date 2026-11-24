#cd ~/LIORA_PROJET_PATHO_RESPIRATOIRES && source venv/bin/activate && cd Streamlit/LIORA && streamlit run project_streamlit.py
# Lancement : streamlit run project_app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic – Prédiction", page_icon="🚢", layout="wide")
st.title("🚢 Survie sur le Titanic")
st.caption("Modèles pré-entraînés chargés depuis modeles_titanic.joblib")

# ---------- Chargements mis en cache ----------
@st.cache_resource          # objets lourds (modèles) : chargés une seule fois
def load_models():
    return joblib.load("modeles_titanic.joblib")

@st.cache_data              # données : mises en cache elles aussi
def load_data():
    return pd.read_csv("train.csv")

modeles = load_models()
df = load_data()

# ---------- Barre latérale : choix du modèle ----------
with st.sidebar:
    st.header("Modèle")
    choix = st.radio("Choisissez un modèle", list(modeles.keys()))
    st.metric("Accuracy (jeu de test)", f"{modeles[choix]['accuracy']:.1%}")

tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Données"])

# ---------- Prédiction à partir des saisies ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    pclass = c1.selectbox("Classe", [1, 2, 3], index=2)
    sex = c1.radio("Sexe", ["male", "female"], horizontal=True)
    age = c2.slider("Âge", 0, 80, 30)
    fare = c2.number_input("Prix du billet (£)", 0.0, 520.0, 15.0)
    sibsp = c3.number_input("Frères/sœurs/conjoint à bord", 0, 8, 0)
    parch = c3.number_input("Parents/enfants à bord", 0, 6, 0)
    embarked = c3.selectbox("Port d'embarquement", ["S", "C", "Q"])

    # Une ligne de données BRUTES : le pipeline fait le prétraitement
    passager = pd.DataFrame([{
        "Age": age, "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Pclass": pclass, "Sex": sex, "Embarked": embarked,
    }])

    proba = modeles[choix]["pipeline"].predict_proba(passager)[0, 1]
    st.metric("Probabilité de survie", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= 0.5:
        st.success("Survit probablement")
    else:
        st.error("Ne survit probablement pas")

# ---------- Exploration des données d'entraînement ----------
with tab2:
    st.dataframe(df.head(20), use_container_width=True)
    var = st.selectbox("Taux de survie selon", ["Sex", "Pclass", "Embarked"])
    st.bar_chart(df.groupby(var)["Survived"].mean())