# Databricks notebook source
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

print("✅ Imports OK")

# COMMAND ----------

# Leer desde Unity Catalog (no /mnt/)
nyctaxi_silver = spark.read.table("bigdata_final.taxis.silver")

df = nyctaxi_silver.select(
    "fare_per_mile", "time_period", "trip_type",
    "pickup_hour", "pickup_dayofweek", "vendor_id",
    "payment_type", "passenger_count"
).dropna().sample(fraction=0.82, seed=42).toPandas()

print(f"📦 Registros: {len(df):,}")

# COMMAND ----------

p33 = df["fare_per_mile"].quantile(0.33)
p66 = df["fare_per_mile"].quantile(0.66)

df["fare_label"] = df["fare_per_mile"].apply(
    lambda x: 0 if x <= p33 else (1 if x <= p66 else 2)
)

cat_cols = ["time_period", "trip_type", "vendor_id", "payment_type"]
num_cols = ["pickup_hour", "pickup_dayofweek", "passenger_count"]

for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))

print(f"🏷️  Low: <${p33:.2f} | Mid: ${p33:.2f}–${p66:.2f} | High: >${p66:.2f}")
print(df["fare_label"].value_counts().sort_index())

# COMMAND ----------

# DBTITLE 1,Cell 4
from sklearn.model_selection import train_test_split

X = df[cat_cols + num_cols]
y = df["fare_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🏋️  Train: {len(X_train):,} | 🧪 Test: {len(X_test):,}")

mlflow.set_experiment("/Users/0234976@up.edu.mx/dynamic_pricing")

with mlflow.start_run(run_name="LR_sklearn"):
    lr = LogisticRegression(
        max_iter=500,
        C=5.0,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("model", "LogisticRegression_sklearn")
    mlflow.log_param("C", 5.0)
    mlflow.log_param("max_iter", 500)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    # Create signature for UC registration
    signature = mlflow.models.infer_signature(X_train, lr.predict(X_train))
    input_example = X_train.head(5)
    
    mlflow.sklearn.log_model(
        lr, 
        "model",
        signature=signature,
        input_example=input_example
    )

    print(f"✅ LR sklearn → Accuracy: {acc:.4f} | F1: {f1:.4f}")

# COMMAND ----------

import pickle

with open("/Volumes/bigdata_final/taxis/mlflow_tmp/lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("🏆 Modelo guardado")
print("🏁 ML completo")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"

registered = mlflow.register_model(
    model_uri=model_uri,
    name="bigdata_final.taxis.dynamic_pricing_lr"
)

print(f"✅ Modelo registrado → versión {registered.version}")

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name="bigdata_final.taxis.dynamic_pricing_lr",
    alias="Production",
    version=registered.version
)

print(f"✅ Versión {registered.version} → alias: Production")

# COMMAND ----------

import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

endpoint_url = "https://dbc-253d6731-2274.cloud.databricks.com/serving-endpoints/dynamic_pricing_endpoint/invocations"

data = {
    "dataframe_records": [
        {
            "time_period": 0, "trip_type": 1, "pickup_hour": 8,
            "pickup_dayofweek": 2, "vendor_id": 0,
            "payment_type": 1, "passenger_count": 1
        },
        {
            "time_period": 2, "trip_type": 0, "pickup_hour": 2,
            "pickup_dayofweek": 6, "vendor_id": 1,
            "payment_type": 0, "passenger_count": 2
        },
        {
            "time_period": 1, "trip_type": 3, "pickup_hour": 18,
            "pickup_dayofweek": 1, "vendor_id": 0,
            "payment_type": 1, "passenger_count": 3
        }
    ]
}

response = requests.post(
    endpoint_url,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    },
    json=data
)

predictions = response.json()["predictions"]

print("══ Predicciones Dynamic Pricing ══")
for i, pred in enumerate(predictions):
    print(f"  Viaje {i+1}: {pred}")   # ← ahora pred ya es texto, no número

# COMMAND ----------

# ── RANGOS DE PRECIO (FIXED con percentiles) ───────────────────────
# Usamos percentiles para evitar outliers extremos

p_min = float(df['fare_per_mile'].quantile(0.01))  # Percentil 1
# p33 y p66 ya están definidos desde celda 3
p_max = float(df['fare_per_mile'].quantile(0.99))  # Percentil 99 (no máximo absoluto)

print("=" * 60)
print("💰 RANGOS DE TARIFA (fare per mile) - ACTUALIZADO")
print("=" * 60)
print(f"  🟢 Tarifa Baja:  ${p_min:.2f}/mi  →  ${p33:.2f}/mi")
print(f"  🟡 Tarifa Media: ${p33:.2f}/mi  →  ${p66:.2f}/mi")
print(f"  🔴 Tarifa Alta:  ${p66:.2f}/mi  →  ${p_max:.2f}/mi")
print("=" * 60)
print(f"\n✅ Usando percentiles para evitar outliers")
print(f"   (cubre 99% de los datos reales)")

# COMMAND ----------

import mlflow.pyfunc

# ── WRAPPER ───────────────────────────────────────────────────────────
class PricingWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model, p_min, p33, p66, p_max):
        self.model = model
        self.labels = {
            0: f"Tarifa Baja  (${p_min:.2f} - ${p33:.2f})",
            1: f"Tarifa Media (${p33:.2f} - ${p66:.2f})",
            2: f"Tarifa Alta  (${p66:.2f} - ${p_max:.2f})"
        }

    def predict(self, context, model_input):
        preds = self.model.predict(model_input)
        return [self.labels[int(p)] for p in preds]


# Crear el wrapper con el modelo ya entrenado (lr = tu modelo actual)
wrapper = PricingWrapper(lr, p_min, p33, p66, p_max)

# ── REGISTRAR EN MLFLOW ───────────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = "bigdata_final.taxis.dynamic_pricing_lr"

with mlflow.start_run(run_name="dynamic_pricing_con_etiquetas"):
    # Infer signature from training data
    signature = mlflow.models.infer_signature(X_train, wrapper.predict(None, X_train))
    input_example = X_train.head(5)

    mlflow.pyfunc.log_model(
        artifact_path         = "model",
        python_model          = wrapper,
        signature             = signature,
        input_example         = input_example,
        registered_model_name = MODEL_NAME
    )

    print("✅ Modelo con etiquetas registrado en MLflow")
    print(f"\nModelo: {MODEL_NAME}")
    print(f"\nEtiquetas configuradas:")
    for k, v in wrapper.labels.items():
        print(f"  {k} → {v}")

# COMMAND ----------

# ── PRUEBA LOCAL DEL WRAPPER ──────────────────────────────────────────
import pandas as pd

# Simular el mismo input que mandas desde Postman
# Columns must match training order: cat_cols + num_cols
test_input = pd.DataFrame([{
    "time_period": 0,
    "trip_type": 1,
    "vendor_id": 0,
    "payment_type": 1,
    "pickup_hour": 8,
    "pickup_dayofweek": 2,
    "passenger_count": 1
}])

resultado = wrapper.predict(None, test_input)
print("🧪 Prueba local del wrapper:")
print(f"   Input:  time_period=0, trip_type=1, pickup_hour=8")
print(f"   Output: {resultado[0]}")
print("\n✅ El wrapper funciona correctamente")

# COMMAND ----------

# ── ANÁLISIS DE OUTLIERS EN fare_per_mile ──────────────────────────
print("=" * 60)
print("📊 ANÁLISIS DE fare_per_mile")
print("=" * 60)

# Estadísticas básicas
print(f"\nMín:     ${df['fare_per_mile'].min():.2f}")
print(f"P25:     ${df['fare_per_mile'].quantile(0.25):.2f}")
print(f"P50:     ${df['fare_per_mile'].quantile(0.50):.2f}")
print(f"P75:     ${df['fare_per_mile'].quantile(0.75):.2f}")
print(f"P99:     ${df['fare_per_mile'].quantile(0.99):.2f}")  # ← percentil 99
print(f"Máx:     ${df['fare_per_mile'].max():.2f}")

# Contar outliers
outliers_gt_20 = (df['fare_per_mile'] > 20).sum()
outliers_gt_100 = (df['fare_per_mile'] > 100).sum()

print(f"\nValores > $20/mi:   {outliers_gt_20:,} ({outliers_gt_20/len(df)*100:.2f}%)")
print(f"Valores > $100/mi:  {outliers_gt_100:,} ({outliers_gt_100/len(df)*100:.2f}%)")

# Cuál es un rango razonable
p99 = df['fare_per_mile'].quantile(0.99)
print(f"\n✅ Percentil 99 (cubre 99% de casos): ${p99:.2f}/mi")