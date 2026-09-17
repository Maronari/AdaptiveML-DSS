<div align="center">
[🇷🇺 Русский](README.md) · [🇬🇧 English](README_en.md) · [🇨🇳 中文](README_ch.md)

<h1>Platform for Automated Machine Learning and Decision Support for Complex Systems with Adaptive Regression Retraining on New Actual Data</h1>

<p>
<b>AdaptiveML DSS</b> is a modular platform for automated machine learning (AutoML) and decision
support (DSS) for regression and tabular-data forecasting. It combines LightAutoML-based AutoML,
model version registry, local forecast interpretation with SHAP, Experta expert rules, and
adaptive retraining on new labeled data.
</p>

<p>
<a href="https://github.com/Maronari/AdaptiveML-DSS"><img src="https://img.shields.io/badge/GitHub-AdaptiveML--DSS-lightgrey?style=flat-square&logo=github" alt="GitHub"></a>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License">
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
</p>

<p style="font-size: 1.1em;">
<a href="README.md"><b>Русская локализация</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="README_en.md"><b>English Localization</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="README_ch.md"><b>中文本地化</b></a>
</p>
</div>

<p align="center">
<b>A. E. Dzgoev, D. S. Kolganov, I. V. Korsunov, I. A. Filonov</b><br>
<i>MIREA — Russian Technological University</i><br>
<i>78 Vernadsky Avenue, Moscow, 119454, Russian Federation</i>
</p>

<h2 align="center">Scientific Collaboration</h2>

<p>The authors present AdaptiveML DSS as a basis for research and practical applications of
automated machine learning, forecast interpretation, and intelligent decision support in
energy, industry, data analysis, and other domains.</p>

<h2 align="center">Citation</h2>

A. E. Dzgoev, D. S. Kolganov, I. V. Korsunov, I. A. Filonov.
<i>Platform for Automated Machine Learning and Decision Support for Complex Systems with Adaptive Regression Retraining on New Actual Data.</i>

<h2 align="center">Abstract</h2>

Modern automated machine learning systems for tabular data have limitations related to the lack
of built-in mechanisms for adaptive retraining on new data, local forecast interpretation, and
decision support. AdaptiveML DSS combines LightAutoML-based AutoML, dataset/model versioning,
SHAP-based local interpretation, and an Experta-based decision support system.

The platform supports CSV/Excel upload and validation, automatic tabular preprocessing, model
training and storage, prediction with schema compatibility checks, visualization, and adaptive
retraining. Its key pipeline is <b>forecast → SHAP factors → structured facts → expert rules → recommendation</b>.
Rules are configured in JSON without modifying application code.

Experiments were performed on real electricity-consumption data for 2019–2022. R² was 0.9967
on the training set and 0.9721 on the test set. The retraining mechanism also demonstrates that
a degraded Latest model does not replace the active Champion model.

<b>Keywords:</b> automated machine learning, AutoML, tabular data, SHAP, decision support system,
DSS, LightAutoML, adaptive retraining, model registry, electricity consumption forecasting,
Experta, web interface.

<h2 align="center">INTRODUCTION</h2>

Machine learning methods are widely used for forecasting and analysis of tabular data. Their
operational use is complicated by changing feature and target distributions over time. Data
drift and concept drift can reduce model accuracy, requiring quality control and regular retraining.

A second problem is the “black box” effect: a numerical forecast alone does not explain which
features influenced the result. AdaptiveML DSS addresses this by combining automated model
selection with versioning, controlled retraining, local SHAP explanations, and deterministic
expert rules.

<h2 align="center">1. REVIEW OF EXISTING SOLUTIONS</h2>

 considers commercial AutoML platforms H2O Driverless AI, DataRobot, and Google AutoML Tables,
as well as the open-source frameworks LightAutoML, AutoGluon, and FLAML.

The proposed platform adds an operational layer for model/data versioning, adaptive retraining,
local forecast explanation, and rule-based decision support.

<h2 align="center">2. PLATFORM ARCHITECTURE</h2>

The platform has five main functional components:

1. data upload and validation;
2. AutoML core;
3. model and dataset registry;
4. prediction and explanation services;
5. decision support system.

The implementation includes FastAPI, `DatasetService`, `TrainingService`, `PredictionService`,
`ExplanationService`, `DecisionService`, SQLite, an HTML/CSS/JavaScript web interface, Chart.js,
and Docker deployment.

### 2.1 Data Upload and Validation

CSV and Excel datasets are supported. The platform detects numeric, categorical, and datetime
columns; checks missing values and duplicates; processes time columns; extracts calendar
features; and verifies schema compatibility for prediction.

### 2.2 AutoML Core

LightAutoML performs preprocessing, categorical encoding, algorithm selection, and pipeline
selection on a hold-out set. Candidate algorithms include LightGBM, CatBoost, XGBoost,
RandomForest, linear models, and MLP.

### 2.3 Model Registry

Each model receives a unique identifier, version, and status:

| Status | Meaning |
|---|---|
| **Champion** | best model by the main quality metric and default production model |
| **Candidate** | model trained on new data and awaiting quality evaluation |
| **Latest** | newest trained version |
| **Archived** | older version retained for history or rollback |

### 2.4 Data Flow

```text
User / Decision Maker
        |
        v
 Web Interface
        |
        v
   FastAPI API
        |
        +--------------------+
        |                    |
        v                    v
 DatasetService        TrainingService
        |                    |
        v                    v
 D1 Dataset Registry    D2 Model Registry
                             |
                             v
                      D3 Artifact Store
                             |
                             v
                  Prediction / Explanation
                             |
                       +-----+-----+
                       |           |
                      SHAP       Forecast
                       |           |
                       +-----+-----+
                             |
                             v
                       DecisionService
                             |
                             v
                       D4 Rule Database
                             |
                             v
                     DSS Recommendation
```

<h2 align="center">3. PREDICTION, INTERPRETATION AND DSS</h2>

The prediction service loads the active Champion model, validates input schema compatibility,
and produces forecasts.

SHAP provides local feature contributions; the interface displays the top 3–5 factors with
numeric influence.

The DSS converts the forecast and SHAP factors into structured facts and applies Experta rules
to produce a recommendation. JSON configuration allows rule sets and trigger conditions to be
changed without modifying source code.  includes `manual_review` and `targeted_diagnostics`
scenarios.

<h2 align="center">4. VISUALIZATION AND WEB INTERFACE</h2>

The interface supports project management, data upload, training, version history, forecasting,
and result analysis. Interactive charts provide zooming, forecast-interval selection, forecasts,
actual values when available, forecast/error tables, SHAP factors, DSS recommendations, and
quality metrics including R², RMSE, AIC, and BIC.

<h2 align="center">5. ADAPTIVE RETRAINING</h2>

Models can be retrained when new labeled actual data arrive.  supports user-triggered retraining
and daily training of candidate models.

A new model does not automatically replace Champion merely because it is the latest version.

| Model ID | Status | RMSE | Change |
|---|---|---:|---:|
| `model-cf9eb8376072` | Archived | 3.1248 | initial model |
| `model-3e4cc357efc5` | Candidate | 2.7803 | −11.0% |
| `model-a36f46663244` | **Champion** | **2.7106** | −2.5% |
| `model-1d6878955cb0` | Latest | 2.9485 | +8.8% |

RMSE improved from 3.1248 to 2.7106 (13.2% relative reduction). The Latest model with RMSE
2.9485 was not activated as Champion because its quality degraded.

<h2 align="center">6. EXPERIMENTAL EVALUATION</h2>

Experiments used real electricity-consumption data for 2019–2022.  identifies the source as
PJSC Rosseti North Caucasus (JSC Sevkavkazenergo).

Metrics include RMSE, MAE, R², AIC, and BIC. Forecasting was also evaluated at 24-, 72-, and
168-hour horizons using MAE, RMSE, MAPE, and R².

| Split | R² |
|---|---:|
| Training | **0.9967** |
| Test | **0.9721** |

The DSS complements the forecast with SHAP explanations and a structured rule-based recommendation.

<h2 align="center">7. PRACTICAL APPLICATION</h2>

AdaptiveML DSS can be used for electricity-consumption forecasting, tabular-data analysis,
industrial forecasting systems, model-quality monitoring, intelligent decision support, and
domains where expert rules can be formalized in JSON.

<h2 align="center">8. CONCLUSION</h2>

 presents a modular platform combining automated regression modeling, version management,
forecasting, SHAP interpretation, Experta-based DSS, and controlled adaptive retraining.

The quality gate prevents a degraded newly trained model from replacing the working Champion.
On real electricity-consumption data, R² reached 0.9967 on training and 0.9721 on test data.

<h2 align="center">9. FUTURE DEVELOPMENT</h2>

- Dask and Apache Spark for datasets larger than one million rows;
- Evidently-based automatic data/concept drift detection;
- drag-and-drop visual rule editor;
- P10/P90 prediction intervals;
- MLflow and Kubeflow integration for continuous deployment and monitoring.

<h2 align="center">REFERENCES</h2>

1. Kustitskaya T.A., Esin R.V., Kytmanov A.A., Zykova T.V. Monitoring data and concept drift in industrial machine learning systems: metrics, quality gates and drift monitoring. Molodoy uchenyy, 2026, no. 3(613).
2. Sobolevsky V.A. Automation of machine learning model creation for time-series forecasting tasks. Izvestiya vuzov. Priborostroenie, 2024, vol. 67, no. 11, pp. 951–957. DOI: 10.17586/0021-3454-2024-67-11-951-957.
3. Kumaritov A.M., Dzgoev A.E., Babochiev O.R., Khuzmiev I.M. Modeling of a cost-control system for decision support in managing a power-grid enterprise. Problemy energetiki, 2015, no. 9–10, pp. 35–43.
4. Biryukov D.N., Suprun A.F. From the “black box” to transparency: philosophical and methodological foundations of explainability and interpretability in artificial intelligence. 2025, no. 1, pp. 30–42. DOI: 10.48612/jisp/x8ve-86ez-fv94.
5. Ageev V.A., Kazakov D.V., Repyev D.S. Review of traditional and neural-network methods for electric-load forecasting. Ogarev-online, 2023.
6. AdaptiveML DSS. GitHub repository. https://github.com/Maronari/AdaptiveML-DSS
7. H2O.ai. H2O Driverless AI — Automated Machine Learning.
8. DataRobot. Enterprise AI Platform — Automated Machine Learning.
9. Google Cloud. AutoML Tables — Automated Machine Learning for Tabular Data.
10. LightAutoML. GitHub repository.
11. AutoGluon. AutoML for Tabular, Text, Image, and Time Series Data.
12. FLAML. A Fast and Lightweight AutoML Library.
13. Alkatsov M.I., Dzgoev A.E., Betrozov M.S. Research and development of a method for forecasting electricity consumption in regional power-supply management. 2012.
14. Khusnutdinov A.O., Khabarov V.I., Karmanov V.S. Deep learning for multivariate time-series analysis. 2025, no. 3 (99), pp. 113–136. DOI: 10.17212/2782-2001-2025-3-113-136.
15. Abotaleb M.S.A. Time-series forecasting algorithms based on weighted least absolute deviations. PhD dissertation. Chelyabinsk, 2024.

<h2 align="center">ADDITIONAL INFORMATION</h2>

### Conflict of Interest
The  text does not provide a separate conflict-of-interest statement; no unsupported statement is added here.

### Authors
A. E. Dzgoev, D. S. Kolganov, I. V. Korsunov, and I. A. Filonov, MIREA — Russian Technological University.

<h2 align="center">LICENSE</h2>

The project is distributed under the <b>Apache License 2.0</b>.
