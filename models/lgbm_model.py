import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

def train_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="LightGBM"):
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        signature = infer_signature(X_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.lightgbm.log_model(model, "lightgbm", signature=signature, input_example=X_test[:1])

        return model
