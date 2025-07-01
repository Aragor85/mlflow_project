import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

def train_use(config, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="USE + MLPClassifier", nested=True):
        # === Standardisation des données ===
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === Initialisation du modèle MLP ===
        mlp_params = config.get("mlp_params", {})
        base_model = MLPClassifier(**mlp_params)

        # === Calibration des probabilités ===
        model = CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv=3)

        # === Entraînement ===
        model.fit(X_train_scaled, y_train)

        # === Prédictions ===
        y_pred = model.predict(X_test_scaled)
        y_proba_all = model.predict_proba(X_test_scaled)  

        # === Évaluation ===
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba_all[:, 1])  
        lloss = log_loss(y_test, y_proba_all)          

        # === Logs MLflow ===
        mlflow.log_params(mlp_params)
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "log_loss": lloss
        })

        # === Sauvegarde du modèle et du scaler ===
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")

        print(f"✅ USE + MLP terminé avec accuracy={acc:.2f} | F1={f1:.2f} | AUC={auc:.2f} | log_loss={lloss:.4f}")
