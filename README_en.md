<div align="center">

<h1>
Automated Machine Learning and Decision Support Platform for Managing
Complex Systems with an Adaptive Retraining Method for Regression on
New Actual Data
</h1>
<p>
<b>AdaptiveML DSS</b> — a modular platform for automated machine learning
(AutoML) and decision support (DSS) for regression and tabular data
forecasting tasks. The platform combines automated training based on
LightAutoML, local interpretation of forecasts using SHAP, a registry of
dataset and model versions, Experta expert rules, and an adaptive
retraining mechanism on new labeled data.
</p>
<p>
<a href="https://github.com/Maronari/AdaptiveML-DSS">
<img src="https://img.shields.io/badge/GitHub-AdaptiveML--DSS-lightgrey?style=flat-square&logo=github" alt="GitHub">
</a>   

<img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
</p>
<p style="font-size: 1.1em;">
<a href="README.md"><b>Russian
Localization</b></a>   \|  
<a href="README_en.md"><b>English
Localization</b></a>   \|  
<a href="README_ch.md"><b>Chinese Localization</b></a>
</p>

</div>

<p align="center">
<b>A. E. Dzgoev, D. S. Kolganov, I. V. Korsunov, I. A.
Filonov</b> <br> <i>MIREA — Russian Technological
University</i> <br>
<i>119454, Russia, Moscow, Vernadsky Avenue,
78</i>
</p>

<h2 align="center" style="border-bottom: none; border: none;">
For Citation
</h2>

&emsp;&emsp;Dzgoev A. E., Kolganov D. S., Korsunov I. V., Filonov I. A.
<i>Automated Machine Learning and Decision Support Platform for Managing
Complex Systems with an Adaptive Retraining Method for Regression on
New Actual Data.</i>

<h2 align="center" style="border-bottom: none; border: none;">
Abstract
</h2>

&emsp;&emsp;Modern automated machine learning (AutoML) systems for tabular data,
despite their active development, have a number of limitations: the
absence of built-in mechanisms for automatic model retraining on new
data, and a lack of tools for local interpretation of forecasts and
decision support. As a result, existing open-source solutions may not
adapt sufficiently to changing data, and the user receives a forecast
without an explanation of how it was formed.

&emsp;&emsp;To assess the current state of the field, a review of domestic and
foreign AutoML frameworks was conducted, in particular LightAutoML,
AutoGluon, and FLAML. On this basis, the AdaptiveML DSS platform was
developed, combining AutoML, local interpretation of forecasts via the
SHAP method, and a decision support system in which forecasts are
interpreted according to a set of expert rules. A key feature is the
adaptive retraining mechanism, which allows models to be updated when
new labeled data arrives, with quality control.

&emsp;&emsp;The experimental part was performed on real electricity consumption
data for 2019–2022. R², RMSE, MAE, AIC, and BIC are used to assess
quality. R² = 0.9967 was obtained on the training set and R² = 0.9721 on
the test set. The decision support system generates recommendations
explaining which features influenced the forecast and how they relate
to the selected scenario.

<b>Keywords:</b> automated machine learning, tabular data, SHAP,
decision support system, LightAutoML, adaptive retraining, model
registry, electricity consumption forecasting, Experta, web interface.

<h2 align="center" style="border-bottom: none; border: none;">
INTRODUCTION
</h2>

&emsp;&emsp;Modern machine learning methods are widely used for forecasting and
classification of tabular data in the energy, industrial, and financial
sectors. However, the practical operation of ML models involves two
significant problems.

&emsp;&emsp;The first problem is related to the phenomenon of data drift
(concept drift / data drift) [1]. The statistical characteristics of
features X or the target variable Y may change under the influence of
seasonality, changes in equipment operating modes, and economic
conditions. As a result, a model trained on a particular sample loses
accuracy over time. Maintaining forecasting quality requires regular
retraining of mathematical models, which in most existing AutoML
solutions is performed manually [2]. For the energy sector, reducing
forecast error has direct practical significance, since it affects the
economic efficiency of power system modes.

&emsp;&emsp;The economic aspect of collecting and processing consumption data is
also important. The costs of data transmission in automated metering
and control systems (AMR) depend on the categories of the metering
devices being polled; metering points of the "small-motor" sector have
a significant impact on financial costs [3]. Therefore, an integrated
approach is advisable, combining forecasting and decision support that
takes into account not only forecast quality but also associated
operational costs.

&emsp;&emsp;The second problem is the non-interpretability of modern models [4],
often called the "black box" effect. Even high forecast accuracy does
not allow the user to understand which features and in what way
influenced the result. The lack of transparency reduces trust in the
AutoML system and hinders informed decision-making based on its outputs.

&emsp;&emsp;Existing open-source AutoML frameworks, in particular LightAutoML,
provide automatic selection of algorithms and hyperparameters but do
not contain built-in mechanisms for retraining on new data, do not
maintain a registry of model versions, and do not contain built-in
tools for interpreting forecasts [5].

&emsp;&emsp;This paper presents the AdaptiveML DSS platform [6], developed to
eliminate these limitations. The platform implements loading and
validation of tabular datasets in CSV and Excel formats; automatic
training of models based on LightAutoML with storage of dataset and
model versions in the registry; predictions on new data with schema
compatibility checking; local explanation of forecasts using SHAP;
generation of recommendations based on Experta expert rules; graphical
visualization with scaling and selection of the forecasting interval;
retraining of models when new labeled data arrives.

&emsp;&emsp;A key feature of the platform is the DSS, which transforms the
forecast and the identified key influence features into a structured
set of facts, after which expert rules based on Experta form the final
recommendation. Rules can be edited via JSON configuration without
changing the program code, which ensures adaptation to various subject
areas.

&emsp;&emsp;The architecture is built on a modular principle and includes a
FastAPI backend, a service layer with DatasetService, TrainingService,
PredictionService, ExplanationService, and DecisionService, a registry
of datasets and models with metadata stored in SQLite, and a web
interface for managing projects, launching training, viewing the
history of model versions, and visualizing forecasts.

<h2 align="center" style="border-bottom: none; border: none;">
1. REVIEW OF EXISTING SOLUTIONS
</h2>

&emsp;&emsp;Automated machine learning systems for regression data can be divided
into commercial platforms and open-source frameworks.

&emsp;&emsp;Commercial solutions include H2O Driverless AI [7], DataRobot [8],
and Google AutoML Tables [9]. They provide automatic selection of
models and hyperparameters, have a developed user interface, and
support basic explainability at the level of global feature importance.
At the same time, they are noted for closed source code, the absence of
mechanisms for automatic retraining when new data arrives, limited
model versioning capabilities, and the need for licensing fees, which
is especially significant for scientific and educational projects.

&emsp;&emsp;Among open-source solutions, LightAutoML [10], AutoGluon [11], and
FLAML [12] stand out. LightAutoML, developed at Sberbank of Russia,
is a lightweight framework for automated training on tabular data with
support for gradient boosting, random forest, linear models, and neural
networks, as well as automatic preprocessing. AutoGluon from Amazon and
FLAML from Microsoft offer similar basic functionality. However, these
solutions do not provide a built-in mechanism for retraining on new
data, a registry of dataset and model versions, and local
explainability of each forecast by default; such functions require
additional integration [13].

&emsp;&emsp;In parallel, in the field of electricity consumption forecasting,
data processing methods are being developed that allow continuous
updating of regression model coefficients by removing outdated data and
adding new actual observations. Since electricity consumption is a
complex multidimensional time series, the systematization of neural
network architectures that take into account temporal dependencies and
structural features of noisy data is becoming increasingly important
[14]. The described approaches note a reduction in forecast error to
4–6 % and an improvement in the quality of power supply management
[15].

#### Table 1. Comparison of LightAutoML and AdaptiveML DSS

| Criterion | LightAutoML | AdaptiveML DSS |
|---|---|---|
| AutoML (tabular data) | Yes | Yes |
| Local explanation (SHAP) | No (manual connection required) | Yes |
| Retraining on new data | No | Yes (at user request) |
| Model versioning | No | Yes |
| Dataset registry | No | Yes |
| Decision support system | No | Yes (Experta) |

&emsp;&emsp;Thus, existing open-source solutions provide basic AutoML
functionality but do not provide an integrated toolkit for maintaining
model relevance over time, managing versions, and local interpretation
of results. AdaptiveML DSS combines AutoML, local explainability with
an assessment of each feature's contribution, a version registry, and a
decision support system based on expert rules.

<h2 align="center" style="border-bottom: none; border: none;">
2. PLATFORM ARCHITECTURE
</h2>

&emsp;&emsp;The AdaptiveML DSS platform is built on a modular principle and
includes a data loading and validation module, an AutoML core, a model
registry, a prediction and explanation service, and a decision support
system. The overall architecture of the components is shown in Fig. 1.

<img src="images/01_architecture_c4.png" alt="Fig. 1. AdaptiveML DSS platform components" width="100%">
<p>
<b>Fig. 1.</b> AdaptiveML DSS platform components (C4
Level 3)
</p>

&emsp;&emsp;The loading and validation module accepts CSV and Excel. The system
automatically determines column types — numeric, categorical, and
date/time — checks for missing values and duplicates, and during
forecasting controls the compatibility of the input data schema. From
temporal columns, features of hour, day of week, and month are
extracted.

&emsp;&emsp;The AutoML core is implemented on the basis of LightAutoML. It
performs automatic preprocessing, including filling in missing values
and encoding categorical variables, iterates over the algorithms
LightGBM, CatBoost, XGBoost, RandomForest, linear models, and neural
networks, including a multilayer perceptron (MLP), and then selects the
best pipeline on a hold-out set. The trained model is serialized and
saved together with meta-information.

&emsp;&emsp;The model registry stores dataset and model versions. Each model is
assigned a unique identifier, version, and status:
<b>Candidate</b>, <b>Champion</b>,
<b>Latest</b>, or <b>Archived</b>.
Model artifacts — serialized models and preprocessors — are
saved together with meta-information.

| Status | Purpose |
|---|---|
| **Champion** | the model recognized as the best by the main quality metric (RMSE for regression) and used for forecasts by default |
| **Candidate** | a candidate model trained on new data but not surpassing the Champion |
| **Latest** | the most recently trained version |
| **Archived** | an outdated model saved for the possibility of returning to it, for example for retraining |

&emsp;&emsp;Such a status system ensures transparency of retraining and allows
the user to return to previous versions. The prediction and explanation
service loads the active Champion model, checks the compatibility of
input data, and performs forecasting. For local explanation, SHAP
values are used; the user is presented with 3–5 of the most significant
factors with an indication of their numerical influence.

&emsp;&emsp;The DSS transforms the forecast and SHAP factors into a structured
set of facts, after which Experta forms the final recommendation. Rules
are edited via JSON configuration without changing the code. The
possibility of daily retraining of candidate models when new actual
data arrives is provided, which allows adaptation to changing
conditions without full retraining from scratch.

&emsp;&emsp;The scenario of interaction between components when processing a user
request is shown in Fig. 2.

<img src="images/02_service_interaction_sequence.png" alt="Fig. 2. UML sequence diagram" width="100%">
<p>
<b>Fig. 2.</b> UML sequence diagram:
user request processing scenario
</p>

&emsp;&emsp;The diagram reflects the sequence of calls between the web
interface, API Gateway, and the service layer, as well as the return of
model metrics, forecast values, SHAP factors, and final recommendations.

&emsp;&emsp;Data flows between processes and storages are detailed in Fig. 3.
The external entities are the user and the decision maker (DM). The
main processes include data loading and validation, automatic training,
forecasting, SHAP interpretation, and recommendation generation. Data
storages: D1 — dataset registry, D2 — model registry, D3 — artifact
storage, D4 — DSS rule base.

<img src="images/03_data_flow_diagram.png" alt="Fig. 3. Data flow diagram" width="700">
<p>
<b>Fig. 3.</b> Data flow diagram (DFD)
of the AdaptiveML DSS platform
</p>

&emsp;&emsp;The web interface is a single-page application in HTML/CSS/JS using
Chart.js. The user has access to loading datasets, launching training,
viewing the history of model versions, an interactive forecast graph
with scaling and period selection, as well as displaying
recommendations. All components are integrated via REST API based on
FastAPI. The backend and frontend are containerized using Docker.

<h2 align="center" style="border-bottom: none; border: none;">
3. VISUALIZATION AND INTERPRETATION OF FORECASTS
</h2>

&emsp;&emsp;The platform provides two main interfaces for working with
forecasts: a graphical forecast display page and a decision support
system page. Both interfaces use the same model version, which ensures
consistency of the displayed data.

&emsp;&emsp;On the visualization page, an interactive forecast graph for a
specified future time interval is available. You can select the
forecast period, for example from 1 to 24 hours or 1 day, and adjust
the number of displayed points. The graph supports mouse-wheel scaling
and panning by dragging. When hovering the cursor, the exact forecast
value is displayed. For the model version, R², RMSE, AIC, and BIC are
output. Forecasts are also presented in a table with a timestamp and a
numerical value for each step.

<img src="images/04_forecast_dashboard.png" alt="Fig. 4. Forecast graph for model model-a36f46663244" width="100%">
<p>
<b>Fig. 4.</b> Forecast graph for model
<code>model-a36f46663244</code> (Champion)
</p>

&emsp;&emsp;The forecast interval in Fig. 4 is 1 day: 24 points with a step of
1 hour. Historical actual values, model-calculated values on history,
the forecast for the future period, and actual values after the
forecast interval are displayed. The ordinate axis shows electricity
consumption in MWh, the abscissa axis shows time. Comparison of the
forecast and actual lines allows visual assessment of forecasting
accuracy.

&emsp;&emsp;For quantitative assessment of accuracy, a table of forecast points
with time, forecast and actual values, absolute and relative error is
used. The absolute error is defined as

$$\Delta_i = |y_i - \hat{y}_i|, \text{ MWh}.$$

The relative error is defined as

$$\delta_i = \frac{|y_i-\hat{y}_i|}{|y_i|}\cdot100\%.$$

With an actual value of 166.065 MWh and a forecast of 161.703 MWh, the
absolute error is 4.362 MWh, the relative error is 2.7 %.

<img src="images/05_model_metadata.png" alt="Model metadata and quality criteria interface" width="100%">
<p>
<b>Fig. 5.</b> Model metadata and quality criteria
interface
</p>

&emsp;&emsp;Each trained model is identified by a unique string identifier in
the format
<code>model-xxxxxxxxxxxx</code> (12 characters), which
is automatically generated when the version is saved. This allows
linking forecasts, metrics, and datasets to a specific model version
and tracking quality changes during retraining.

&emsp;&emsp;For model <code>model-a36f46663244</code>
(Champion), the following are indicated: LightAutoML (LAMA) /
TabularAutoML algorithm, actual ensemble composition — LightGBM, target
variable — "Electricity consumption", 7 features, training dataset
<code>dataset-03afb6540a8f</code> (13,867 rows),
backend <code>lightautoml</code>, preset
<code>tabular</code>, algorithms
<code>lgb</code> and
<code>linear_l2</code>. This information is saved in the
model metadata and is available to the user through the quality
criteria interface.

&emsp;&emsp;The DSS page is intended for interpretation of forecasts and
generation of recommendations. The user sees a table of forecasts with
step, time, and numerical value indicated, and for each forecast the
system calculates the most significant factors and their numerical
contribution.

&emsp;&emsp;To quantify the contribution of the $j$-th feature, the SHAP method
(SHapley Additive exPlanations) is used. The Shapley value for feature
$j$ is defined as

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[f(S\cup\{j\})-f(S)\right],$$

where $F$ is the set of all features, $S$ is a subset of features,
$f(S)$ is the model forecast when using only features from the set $S$.
The value $\phi_j$ is interpreted as the contribution of the $j$-th
feature to the deviation of the forecast from the mean value [16].

&emsp;&emsp;For electricity consumption data, the factors considered are
"exceeding the 90th percentile of historical values", "deviation from
the last actual value", "deviation from the average level", and
"falling into peak hours". The numerical values of the factors may, for
example, be 2.1118; 1.3488; 1.3022; 1.0, which allows quantifying the
degree of their influence.

&emsp;&emsp;Recommendation generation is performed on the basis of
deterministic expert rules defined in the JSON configuration. For each
forecast point, based on the forecast and SHAP factors, a set of facts
is constructed — risk level, strong positive and constraining factors.
The set of facts is matched against Experta rules. When conditions
match, a scenario is selected, from which the final recommendation, a
list of actions, and a justification are extracted. This approach
ensures reproducibility of results, excludes uncontrolled text
generation, and allows the DM to understand the logic of each
recommendation conclusion.

<img src="images/06_dss_recommendation_panel.jpeg" alt="Fig. 6. Decision support system interface" width="100%">
<p>
<b>Fig. 6.</b> Decision support system
interface
</p>

&emsp;&emsp;For a forecast point with a value of 180.07 MWh (step 2), the rules
determine a high risk level (HIGH) based on three strong positive
factors: "deviation from the average level", "deviation from the last
actual value", and "falling into peak hours". The scenario
<code>load_shedding</code> is selected with a recommendation
to reduce load for the nearest forecast window.

&emsp;&emsp;For a forecast with a value of 140.28, the system generates
recommendations to "continue observation without immediate
intervention" and to "collect additional data to track changes in the
nature of the data". For each forecast step, the user sees the degree
of confidence, recommended actions for the DM, and a detailed
justification.

&emsp;&emsp;DSS rules can be edited directly in the web interface via JSON
configuration. The user can change rule sets, scenarios, and triggering
conditions without changing the code. The configuration may define
scenarios
<code>manual_review</code> with actions to reduce load
and assign diagnostics, as well as
<code>targeted_diagnostics</code>. This ensures
adaptation of the system to various subject areas and business
requirements.

&emsp;&emsp;Thus, the visualization and interpretation subsystem allows the user
not only to obtain forecasts but also to understand the reasons for the
obtained values, as well as to receive ready-made recommendations for
decision-making.

<h2 align="center" style="border-bottom: none; border: none;">
4. FORECASTING QUALITY ASSESSMENT
</h2>

&emsp;&emsp;To quantitatively assess the quality of regression models, RMSE,
MAE, R², AIC, and BIC are used [17].

<b>Root Mean Square Error (RMSE)</b>
characterizes the magnitude of deviation of forecast values from actual
values:

$$RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2},$$

where $N$ is the sample size, $y_i$ are the actual values, $\hat{y}_i$
are the forecast values.

<b>Mean Absolute Error (MAE)</b> determines
the average absolute deviation:

$$MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|.$$

<b>Coefficient of Determination (R²)</b> shows the
proportion of variance of the dependent variable explained by the
model:

$$R^2=1-\frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{N}(y_i-\bar{y})^2},$$

where $\bar{y}$ is the arithmetic mean of the actual values.

&emsp;&emsp;To compare models with different numbers of parameters and select
the most adequate structure, information criteria are used.

<b>Akaike Information Criterion (AIC)</b>:

$$AIC=2k-2\ln L,$$

where $k$ is the number of model parameters, $L$ is the maximum of the
likelihood function. A lower AIC value is considered preferable.

<b>Bayesian Information Criterion (BIC)</b>:

$$BIC=k\ln N-2\ln L,$$

where $k$ is the number of model parameters, $N$ is the sample size,
$L$ is the maximum of the likelihood function. BIC imposes a stricter
penalty for increasing the number of parameters compared to AIC.

&emsp;&emsp;The metrics allow a comprehensive assessment of the quality of
candidate models, model selection, and comparison of results on the
training and test sets. The output of R², RMSE, AIC, and BIC on the
platform pages has not only diagnostic but also applied significance:
the user can independently compare model versions and make a decision
on operation or additional retraining. The responsibility for the final
decision rests with the decision maker.

&emsp;&emsp;The platform provides tracking of the dynamics of model quality
during retraining. Fig. 7 shows the history of RMSE changes for four
consecutive model versions as new data arrives.

<img src="images/07_model_retraining_history.png" alt="Fig. 7. RMSE dynamics during retraining" width="100%">
<p>
<b>Fig. 7.</b> RMSE dynamics during
retraining
</p>

| Version | Status | RMSE | Change |
|---|---|---:|---|
| `model-cf9eb8376072` | Archived | 3.1248 | initial model |
| `model-3e4cc357efc5` | Candidate | 2.7803 | improvement 0.3445 (11.0 %) |
| `model-a36f46663244` | Champion | 2.7106 | improvement 0.0697 (2.5 %) |
| `model-1d6878955cb0` | Latest | 2.9485 | deterioration 0.2379 (8.8 %) |

&emsp;&emsp;Forecasting quality improved from RMSE 3.1248 to 2.7106, that is,
the reduction relative to the initial model was 13.2 %. The last Latest
model was not activated as Champion due to quality deterioration. This
demonstrates the platform's protective mechanism: a model with worse
indicators does not replace the current Champion.

<h2 align="center" style="border-bottom: none; border: none;">
CONCLUSION
</h2>

&emsp;&emsp;In the course of the study, the AdaptiveML DSS platform was
developed — a specialized automated machine learning system for
regression tasks, combining AutoML, local interpretation of forecasts,
and decision support.

&emsp;&emsp;A modular architecture was designed with data loading and
validation, an AutoML core based on LightAutoML, a registry of dataset
and model versions, a service for calculating forecast estimates and
explanations, and a decision support system. Integration was performed
via REST API based on FastAPI.

&emsp;&emsp;Local interpretability is provided by SHAP integration. For each
forecast point, feature contribution values are calculated and a list
of the most significant factors with quantitative estimates of
influence is provided, including exceeding the 90th percentile,
deviation from the last actual value, and falling into peak hours.

&emsp;&emsp;The DSS is implemented on the basis of Experta: the forecast and
SHAP factors are transformed into a structured set of facts, after
which recommendations are formed. Rules are edited via JSON
configuration without modifying the program code.

&emsp;&emsp;A web interface was developed with an interactive forecast graph,
scaling, selection of the forecast interval, display of R², RMSE, AIC,
and BIC, and an interface for viewing recommendations.

&emsp;&emsp;The platform's operability was experimentally confirmed on real
electricity consumption data for 2019–2022. The data source is
indicated as the energy company PJSC Rosseti North Caucasus
(JSC Sevkavkazenergo). The trained model demonstrates R² = 0.9967 on
the training set and R² = 0.9721 on the test set. The DSS generates
justified recommendations based on the calculated forecast factors.

&emsp;&emsp;The platform can be used in forecasting, data analysis, and
intelligent decision support systems.

<h2 align="center" style="border-bottom: none; border: none;">
FURTHER DEVELOPMENT DIRECTIONS
</h2>

1. Scaling the processing of datasets with more than one million rows
   using **Dask** and **Apache Spark**.

2. Introduction of an automatic data drift detector based on
   **Evidently** to initiate retraining without user involvement.

3. Development of a visual rule editor with a graphical interface of
   the **drag-and-drop** type.

4. Construction of forecast corridors based on **P10** and **P90**
   quantiles with graphical visualization.

5. Integration with industrial MLOps tools **MLflow** and **Kubeflow**
   for continuous deployment and monitoring of models.

<h2 align="center" style="border-bottom: none; border: none;">
REFERENCES
</h2>

1. Kustitskaya T. A., Esin R. V., Kytmanov A. A., Zykova T. V.
   Monitoring of data and concept drift in industrial machine learning
   systems: metrics, quality gates, and drift monitoring // <i>Young
   Scientist</i>. 2026. No. 3(613). URL:
   https://moluch.ru/archive/613/134123 (accessed: 15.08.2026).

2. Sobolevsky V. A. Automation of creating machine learning models for
   solving time series forecasting problems //
   <i>Izvestiya Vuzov. Instrument Making</i>. 2024.
   Vol. 67. No. 11. Pp. 951–957. DOI:
   10.17586/0021-3454-2024-67-11-951-957. URL:
   https://old-pribor.itmo.ru/ru/article/23165/avtomatizaciya-sozdaniya-modeley-mashinnogo-obucheniya-dlya-resheniya-zadach-prognozirovaniya-vremennyh-ryadov.htm
   (accessed: 15.08.2026).

3. Kumaritov A. M., Dzgoev A. E., Babochiev O. R., Khuzmiev I. M.
   Modeling of a cost control system for decision support in managing
   an electric grid enterprise // <i>Energy Problems</i>. 2015.
   No. 9–10. Pp. 35–43.

4. Biryukov D. N., Suprun A. F. From the "black box" to transparency:
   philosophical and methodological foundations of explainability and
   interpretability in artificial intelligence // <i>Information
   Security Problems. Computer Systems</i>. 2025. No. 1. Pp.
   30–42. DOI: 10.48612/jisp/x8ve-86ez-fv94.

5. Ageev V. A., Kazakov D. V., Repiev D. S. Review of traditional and
   neural network methods for electric load forecasting //
   <i>Ogarev-online</i>. 2023.

6. AdaptiveML DSS. GitHub repository. URL:
   https://github.com/Maronari/AdaptiveML-DSS

7. H2O.ai. H2O Driverless AI — Automated Machine Learning. URL:
   https://www.h2o.ai/products/h2o-driverless-ai/

8. DataRobot. Enterprise AI Platform — Automated Machine Learning.
   URL: https://www.datarobot.com/ (accessed: 15.08.2026).

9. Google Cloud. AutoML Tables — Automated Machine Learning for
   Tabular Data. URL: https://cloud.google.com/automl-tables (accessed:
   15.08.2026).

10. LightAutoML. GitHub repository. URL:
    https://github.com/sberbank-ai/LightAutoML (accessed: 15.08.2026).
    Documentation: https://lightautoml.readthedocs.io/

11. AutoGluon. AutoML for Tabular, Text, Image, and Time Series Data.
    URL: https://auto.gluon.ai/ (accessed: 15.08.2026). GitHub:
    https://github.com/autogluon/autogluon

12. FLAML. A Fast and Lightweight AutoML Library. URL:
    https://microsoft.github.io/FLAML/ (accessed: 15.08.2026).
    GitHub: https://github.com/microsoft/FLAML

13. Alkatsev M. I., Dzgoev A. E., Betrozov M. S. Research and
    development of a method for forecasting electricity consumption in
    a regional power supply management system // <i>News of Higher
    Educational Institutions. Energy Problems</i>. 2012. No. 5–6.

14. Khusnutdinov A. O., Khabarov V. I., Karmanov V. S. Deep learning
    for analyzing multidimensional time series: systematization of
    data types, tasks, architectures, and approaches // <i>Data
    Analysis and Processing Systems</i>. 2025. No. 3 (99). Pp. 113–136.
    DOI: 10.17212/2782-2001-2025-3-113-136. URL:
    https://cyberleninka.ru/article/n/glubokoe-obuchenie-dlya-analiza-mnogomernyh-vremennyh-ryadov-sistematizatsiya-tipov-dannyh-zadach-arhitektur-i-podhodov
    (accessed: 15.08.2026).

15. Abotaleb M. S. A. Time series forecasting algorithms based on the
    weighted least absolute deviations method: Cand. Sci. (Phys.-Math.)
    dissertation. Chelyabinsk, 2024. 182 p.

16. Vorobyev A. V. Method for selecting a machine learning model based
    on the stability of predictors using the Shapley value //
    <i>Economics. Informatics</i>. 2021. Vol. 48. No. 2.
    Pp. 350–359. DOI: 10.52575/2687-0932-2021-48-2-350-359. URL:
    https://cyberleninka.ru/article/n/metod-vybora-modeli-mashinnogo-obucheniya-na-osnove-ustoychivosti-prediktorov-s-primeneniem-znacheniya-shepli
    (accessed: 15.08.2026).

17. Dedeveshin A. S., Klyachkin V. N. Influence of sample size on the
    quality of regressions under various machine learning methods //
    <i>Bulletin of Ulyanovsk State Technical University</i>. 2024.
    No. 2 (106). Pp. 37–42. URL:
    https://cyberleninka.ru/article/n/vliyanie-obyoma-vyborki-na-kachestvo-regressiy-pri-razlichnyh-metodah-mashinnogo-obucheniya
    (accessed: 15.08.2026).

<h2 align="center" style="border-bottom: none; border: none;">
AUTHOR INFORMATION
</h2>

- **Alan E. Dzgoev** — Cand. Sci. (Tech.), Associate Professor of the
  Department of Digital Transformation, Institute of Information
  Technologies, MIREA — Russian Technological University. E-mail:
  `Dzgoev@mirea.ru`.

- **Dmitry S. Kolganov** — Master's degree in "Informatics and Computer
  Engineering", Department of Computer Engineering, Institute of
  Information Technologies, MIREA — Russian Technological University.
  E-mail: `dkolganov2000@gmail.com`.

- **Ivan V. Korsunov** — Master's degree in "Informatics and Computer
  Engineering", Department of Computer Engineering, Institute of
  Information Technologies, MIREA — Russian Technological University.
  E-mail: `ivan.corsunov@gmail.com`.

- **Ivan A. Filonov** — Master's degree in "Informatics and Computer
  Engineering", Department of Computer Engineering, Institute of
  Information Technologies, MIREA — Russian Technological University.
  E-mail: `dec200211@gmail.com`.

</div>