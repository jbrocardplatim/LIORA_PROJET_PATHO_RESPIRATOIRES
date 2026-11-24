#cd ~/LIORA_PROJET_PATHO_RESPIRATOIRES && source venv/bin/activate && cd Streamlit/LIORA &&streamlit run project_app.py

# Lancement : python fabrication_titanic.py
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("train.csv")

num_features = ["Age", "SibSp", "Parch", "Fare"]
cat_features = ["Pclass", "Sex", "Embarked"]
X = df[num_features + cat_features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Prétraitement : sera enregistré AVEC le modèle
preproc = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_features),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
])

classifiers = {
    "Régression logistique": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

modeles = {}
for name, clf in classifiers.items():
    pipe = Pipeline([("preproc", clone(preproc)), ("clf", clf)])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    modeles[name] = {"pipeline": pipe, "accuracy": acc}
    print(f"{name} : accuracy test = {acc:.3f}")

# Enregistrement des modèles entraînés
joblib.dump(modeles, "modeles_titanic.joblib")
print("OK : modeles_titanic.joblib écrit")