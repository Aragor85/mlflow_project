import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

def train_use(config, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="USE + MLPClassifier", nested=True):
        # Récupère les paramètres du classifieur MLP
        mlp_params = config.get("mlp_params", {})
        model = MLPClassifier(**mlp_params)

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Évaluation
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        ll = log_loss(y_test, y_proba)

        # Log des hyperparamètres et des métriques
        mlflow.log_params(mlp_params)
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "log_loss": ll
        })

        mlflow.sklearn.log_model(model, "model")

        print(f"✅ USE + MLP terminé avec accuracy={acc:.2f} | F1={f1:.2f} | AUC={auc:.2f} | log_loss={ll:.2f}")
