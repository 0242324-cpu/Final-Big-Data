# 🚕 NYC Taxi — Dynamic Pricing con Medallion Architecture

> Implementación de arquitectura Medallion (Bronze → Silver → Gold) en Databricks con modelo de Machine Learning para predicción de tarifas dinámicas en tiempo real, desplegado como REST API.

---

## 👥 Autores

| Nombre | Institución |
|--------|-------------|
| Angel Guillermo Hernández Arellano | Universidad Panamericana |
| Luis Alejandro Valadez Hernández | Universidad Panamericana |

**Materia:** APRENDIZAJE AUTOMÁTICO PARA GRANDES VOLÚMENES DE DATOS  
**Repositorio:** https://github.com/0242324-cpu/Final-Big-Data

---

## 📋 Descripción

Este proyecto implementa una solución completa de **Big Data + Machine Learning** sobre el dataset público de NYC Taxi (~5 millones de registros). El objetivo principal es construir un pipeline de datos robusto usando la arquitectura Medallion y desplegar un modelo de **Dynamic Pricing** que clasifique cada viaje como tarifa baja, media o alta a través de una REST API consumible en tiempo real.

### Problema que resuelve
Las plataformas de transporte necesitan ajustar tarifas dinámicamente según condiciones del viaje (hora, distancia, tipo de viaje, método de pago). Este proyecto predice el rango de tarifa óptimo para cualquier combinación de parámetros, permitiendo tomar decisiones de pricing en tiempo real.

---

## 🏗️ Arquitectura

```
Datos Crudos (Delta Lake — 5M registros)
            ↓
    ┌─────────────────┐
    │  BRONZE LAYER   │  ← Ingestión sin transformación
    │  bigdata_final  │
    │  .taxis.bronze  │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  SILVER LAYER   │  ← Limpieza + Feature Engineering
    │  bigdata_final  │
    │  .taxis.silver  │
    └────────┬────────┘
             ↓
    ┌─────────────────────────────────────────────────┐
    │                  GOLD LAYER                     │
    │  gold_by_hour  │  gold_by_zone  │  gold_fact    │
    │  gold_by_vendor│  gold_temporal │  gold_payment │
    └────────────────────────┬────────────────────────┘
                             ↓
              ┌──────────────────────────┐
              │     ML MODEL
               (LogisticRegression)      │
              │          │
              └──────────────┬───────────┘
                             ↓
              ┌──────────────────────────┐
              │  MLflow Model Registry   │
              │  dynamic_pricing_lr      │
              │  alias: Production       │
              └──────────────┬───────────┘
                             ↓
              ┌──────────────────────────┐
              │   REST API (Databricks   │
              │   Serving Endpoint)      │
              │   → "Tarifa Alta         │
              │      ($4.76 - $15.71)"   │
              └──────────────────────────┘
```

---

## ⚙️ Stack Tecnológico

| Categoría | Tecnología |
|-----------|-----------|
| **Plataforma** | Databricks (Free Edition) |
| **Almacenamiento** | Delta Lake + Unity Catalog |
| **Procesamiento** | Apache Spark |
| **Catálogo** | Unity Catalog (`bigdata_final.taxis`) |
| **Machine Learning** | scikit-learn, LogisticRegression |
| **Tracking ML** | MLflow (experiments + model registry) |
| **API** | Databricks Serving Endpoints (REST) |
| **Lenguaje** | Python 3.12 |
| **Compute** | Serverless SQL Warehouse 2X-Small |

---

## 📊 Dataset

- **Fuente:** `dbfs:/databricks-datasets/nyctaxi/tables/nyctaxi_yellow` (dataset público de Databricks)
- **Volumen:** 5,000,000 registros
- **Dominio:** Viajes en taxi en la ciudad de Nueva York

### Esquema original (Bronze)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `vendor_id` | string | Proveedor del taxi |
| `pickup_datetime` | timestamp | Fecha/hora de inicio |
| `dropoff_datetime` | timestamp | Fecha/hora de fin |
| `passenger_count` | int | Número de pasajeros |
| `trip_distance` | double | Distancia en millas |
| `pickup_latitude/longitude` | double | Coordenadas de origen |
| `dropoff_latitude/longitude` | double | Coordenadas de destino |
| `payment_type` | string | Método de pago |
| `fare_amount` | double | Tarifa base |
| `total_amount` | double | Monto total cobrado |

---

## 📂 Estructura del Proyecto en Unity Catalog

```
Catalog: bigdata_final
  └── Schema: taxis
        ├── bronze                    ← 5,000,000 registros crudos
        ├── silver                    ← ~4,900,000 registros limpios
        ├── gold_fact_table           ← Tabla de hechos particionada
        ├── gold_by_hour              ← Agregación temporal
        ├── gold_by_zone              ← Agregación geográfica
        ├── gold_by_vendor            ← Métricas por proveedor
        ├── gold_temporal_patterns    ← Patrones de demanda
        ├── gold_payment_analysis     ← Análisis de pagos
        └── [Model Registry]
              └── dynamic_pricing_lr  ← XGBoost con PricingWrapper
                    └── alias: Production
```

---

## 📓 Notebooks

| Notebook | Descripción | Estado |
|----------|-------------|--------|
| `01_bronze_notebook` | Ingestión de 5M registros desde Delta Lake | ✅ |
| `02_silver_notebook` | Limpieza, validación y feature engineering | ✅ |
| `03_gold_notebook` | 7 tablas de agregación para BI y ML | ✅ |
| `04_ml_dynamic_pricing` | Entrenamiento, MLflow, Serving Endpoint | ✅ |

### Pipeline orquestado
```
[01_bronze] → [02_silver] → [03_gold]
```
Ejecutable desde: `Sidebar → Jobs & Pipelines → NYC_Taxi_Medallion_Pipeline → Run now`

---

## 🔄 Capa SILVER — Feature Engineering

A partir de los datos crudos se generaron las siguientes features:

| Feature | Descripción |
|---------|-------------|
| `trip_duration_minutes` | Duración calculada del viaje |
| `pickup_year/month/dayofweek/hour` | Componentes de fecha/hora |
| `fare_per_mile` | Tarifa dividida por distancia |
| `fare_per_minute` | Tarifa dividida por duración |
| `trip_type` | Local / Medium / Long / Very_Long |
| `time_period` | Morning_Rush / Evening_Rush / Night / Off_Peak |

### Filtros de calidad aplicados
- Eliminación de nulos en campos críticos
- Distancia > 0 millas
- Pasajeros entre 1 y 8
- Coordenadas dentro de rangos válidos
- Duración del viaje > 0 minutos

---

## 🎯 Capa GOLD — Tablas de Agregación

| Tabla | Registros aprox. | Uso |
|-------|-----------------|-----|
| `gold_by_hour` | ~8,760 | Demanda por hora del día |
| `gold_by_zone` | ~500 | Hotspots geográficos |
| `gold_by_vendor` | 2 | Comparativa entre vendors |
| `gold_temporal_patterns` | ~1,000 | Patrones semanales/mensuales |
| `gold_payment_analysis` | 4 | Análisis por método de pago |
| `gold_fact_table` | ~4.9M | Tabla desnormalizada para BI |
| `gold_dynamic_pricing` | ~2,000 | Lookup de tarifas recomendadas |

---

## 🧠 Modelo de Machine Learning

### Objetivo
Clasificar cada viaje en uno de tres rangos de tarifa (fare per mile):

| Clase | Etiqueta | Rango |
|-------|---------|-------|
| 0 | 🟢 Tarifa Baja | $0.00 – $3.48 por milla |
| 1 | 🟡 Tarifa Media | $3.48 – $4.76 por milla |
| 2 | 🔴 Tarifa Alta | $4.76 – $15.71 por milla |

> Los rangos están calculados con percentiles (P1–P99) para excluir outliers extremos.

### Modelo: LogisticRegression

```python
XGBClassifier(
    n_estimators     = 200,
    learning_rate    = 0.1,
    max_depth        = 6,
    min_child_weight = 3,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42
)
```

### Features de entrada (7)

| Feature | Tipo | Valores |
|---------|------|---------|
| `time_period` | Categórica | 0=Morning_Rush, 1=Evening_Rush, 2=Night, 3=Off_Peak |
| `trip_type` | Categórica | 0=Local, 1=Medium, 2=Long, 3=Very_Long |
| `vendor_id` | Categórica | 0=Vendor1, 1=Vendor2 |
| `payment_type` | Categórica | 0=Credit, 1=Cash, 2=No Charge, 3=Dispute, 4=Other |
| `pickup_hour` | Numérica | 0–23 |
| `pickup_dayofweek` | Numérica | 1–7 |
| `passenger_count` | Numérica | 1–8 |

### Métricas finales

| Versión | Modelo | Accuracy | F1 |
|---------|--------|----------|----|
| v1 | Logistic Regression | 53.15% | 0.4890 |

---

## 📡 REST API — Serving Endpoint

### Endpoint
```
POST https://dbc-253d6731-2274.cloud.databricks.com/serving-endpoints/dynamic_pricing_endpoint/invocations
```

### Autenticación
```
Authorization: Bearer <DATABRICKS_TOKEN>
Content-Type: application/json
```

> Para generar un token: `Databricks → Settings → Developer → Access tokens → Generate new token`

### Request
```json
{
  "dataframe_records": [
    {
      "time_period": 0,
      "trip_type": 1,
      "pickup_hour": 8,
      "pickup_dayofweek": 2,
      "vendor_id": 0,
      "payment_type": 1,
      "passenger_count": 1
    }
  ]
}
```

### Response
```json
{
  "predictions": [
    "Tarifa Alta  ($4.76 - $15.71)"
  ]
}
```

### Ejemplo en Python
```python
import requests

TOKEN = "dapi_xxxxxxxxxxxx"
URL = "https://dbc-253d6731-2274.cloud.databricks.com/serving-endpoints/dynamic_pricing_endpoint/invocations"

data = {
    "dataframe_records": [{
        "time_period": 0,
        "trip_type": 1,
        "pickup_hour": 8,
        "pickup_dayofweek": 2,
        "vendor_id": 0,
        "payment_type": 1,
        "passenger_count": 1
    }]
}

response = requests.post(
    URL,
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json=data
)

print(response.json()["predictions"][0])
# → "Tarifa Alta  ($4.76 - $15.71)"
```

---

### Request y Response en Postman
![Postman Response](screenshots/postman_response.png)
> Status: 200 OK — Response: `"Tarifa Alta ($4.76 - $15.71)"`

### Rangos de tarifa
```
==================================================
💰 RANGOS DE TARIFA (fare per mile)
==================================================
  🟢 Tarifa Baja:  $0.00/mi  →  $3.48/mi
  🟡 Tarifa Media: $3.48/mi  →  $4.76/mi
  🔴 Tarifa Alta:  $4.76/mi  →  $15.71/mi
==================================================
```

---

## 🔁 Cómo Reproducir el Proyecto

### Pre-requisitos
- Cuenta en Databricks (Free Edition es suficiente)
- Unity Catalog habilitado
- Catálogo `bigdata_final` y schema `taxis` creados

### Paso 1 — Clonar el repositorio
```bash
git clone https://github.com/0242324-cpu/Final-Big-Data
```

### Paso 2 — Importar notebooks a Databricks
```
Workspace → Import → subir archivos .ipynb en orden:
  01_bronze_notebook.ipynb
  02_silver_notebook.ipynb
  03_gold_notebook.ipynb
  04_ml_dynamic_pricing.ipynb
```

### Paso 3 — Ejecutar el pipeline
```
Sidebar → Jobs & Pipelines → NYC_Taxi_Medallion_Pipeline → Run now
```
O ejecutar manualmente en orden: Bronze → Silver → Gold → ML

### Paso 4 — Crear el Serving Endpoint
```
Sidebar → Serving → Create serving endpoint
→ Name: dynamic_pricing_endpoint
→ Model: bigdata_final.taxis.dynamic_pricing_lr
→ Alias: Production
→ Create
```

### Paso 5 — Probar el endpoint
```bash
curl -X POST \
  https://dbc-253d6731-2274.cloud.databricks.com/serving-endpoints/dynamic_pricing_endpoint/invocations \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"time_period": 0, "trip_type": 1, "pickup_hour": 8, "pickup_dayofweek": 2, "vendor_id": 0, "payment_type": 1, "passenger_count": 1}]}'
```

---

## 🔧 Problemas Encontrados y Soluciones

| Problema | Causa | Solución |
|----------|-------|----------|
| Cache overflow con Spark ML | Random Forest con 5M registros excede límite de memoria | Cambiar a scikit-learn / XGBoost en driver |
| `/mnt/` deshabilitado | Public DBFS root desactivado en Free Edition | Migrar a Unity Catalog (`spark.read.table()`) |
| Spark Context no disponible | Serverless no tiene `spark.sparkContext` | Usar `spark.read.table()` + `.toPandas()` |
| Outliers en `fare_per_mile` | Valores hasta $17,000/mi por trips de metros | Usar percentiles P1–P99 en lugar de min/max |
| Endpoint devolvía números (0, 1, 2) | Modelo sin wrapper de etiquetas | Implementar `PricingWrapper` con `mlflow.pyfunc` |

---

## 📈 Estadísticas del Proyecto

| Métrica | Valor |
|---------|-------|
| Registros en Bronze | 5,000,000 |
| Registros en Silver (limpios) | ~4,900,000 |
| Registros usados en ML | 4,063,390 (82% sample) |
| Tablas Gold creadas | 7 |
| Features del modelo | 7 |
| Clases de predicción | 3 (Low / Mid / High) |
| Accuracy del modelo | **53.15%** |
| F1 Score | **0.49** |
| Costo total | **$0** (Databricks Free Edition) |

---






---

*Última actualización: Junio 2026 — Estado: ✅ Completado y en Producción*
