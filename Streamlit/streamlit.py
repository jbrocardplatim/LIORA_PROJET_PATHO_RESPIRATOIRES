import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#cd Streamlit commande pour aller dans le fichier
#streamlit run streamlit.py commande pour runner le streamlit

# --- BARRE LATÉRALE (NAVIGATION) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sommaire", [
    "1.", 
    "2.", 
    "3.", 
    "4.",
    "5."
])

# --- PAGE 1 : CONTEXTE ---
if page == "1.":
    st.title('''Prédiction de l’incidence des pathologies respiratoires''')
    st.image("./images/garde.png")
    st.markdown("""
    Réalisé par: \n 
    Monika LUDANYI, \n 
    Mathilde BOLLECKER, \n 
    Jacques BROCARD, \n 
    Margaux LAGIER \n 
    (#sep26_cds_pathologies_respiratoires)

    Projet suivi par : Romain Lesieur  

    """)

# --- PAGE 2 : EXPLORATION ---
elif page == "2.":
    st.title("")
    st.info("")

# --- PAGE 3 : MODELE ---
elif page == "3.":
    st.title("Modélisation et Évaluation des Performances")

    # Dictionnaire reliant les catégories aux modèles disponibles
    modeles_par_categorie = {
    "Régression ": ["Régression Linéaire", "Random Forest Regressor", "XGBoost Regressor"],
    "Classification ": ["Régression Logistique", "Random Forest Classifier", "XGBoost Classifier"],
    "Séries Temporelles (SARIMA)": ["SARIMA"],
    "Deep Learning": ["DNN (Réseau de Neurones Profond - Régression)", "Bi-LSTM (Exploration)"]
}

    # Création de deux colonnes pour aligner proprement les menus déroulants
    col_cat, col_mod = st.columns(2)

    with col_cat:
        categorie_choisie = st.selectbox(
            "1. Choisir une approche :",
            list(modeles_par_categorie.keys())
    )

    with col_mod:
        modele_choisi = st.selectbox(
            "2. Sélectionner le modèle :",
            modeles_par_categorie[categorie_choisie]
    )

    st.markdown("---")
    
    # Tableau des scores
    st.subheader("Synthèse des performances")
    scores_data = {
        "Jeu de données": ["Train (2021-2022)", "Validation (2023)", "Test (2024)"],
        "R² Score": [0.4944, 0.4446, 0.4034],
        "RMSE": [229.21, 241.06, 226.99],
        "MAE": [165.85, 175.08, 166.06]
    }
    st.table(pd.DataFrame(scores_data))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analyse des résidus")
        st.markdown("titre")
        
        
    with col2:
        st.subheader("Graphique de Parité")
        st.markdown("titre")
        

# --- PAGE 4 : INTERPRÉTABILITÉ ---
elif page == "4.":
    st.title("🔍 Interprétabilité ")
    st.markdown("""
    Texte
    """)