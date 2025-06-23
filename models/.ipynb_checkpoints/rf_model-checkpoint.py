import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

def train_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="RandomForest", nested=True):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Signature pour traçabilité
        signature = infer_signature(X_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "random_forest", signature=signature, input_example=X_test[:1])

        return model
