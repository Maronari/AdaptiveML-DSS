<div align="center">
[🇷🇺 Русский](README.md) · [🇬🇧 English](README_en.md) · [🇨🇳 中文](README_ch.md)

<h1>面向复杂系统的自动化机器学习与决策支持平台：基于新实际数据的自适应回归再训练</h1>

<p>
<b>AdaptiveML DSS</b> 是一个面向回归和表格数据预测的模块化自动化机器学习（AutoML）
与决策支持（DSS）平台。平台结合了基于 LightAutoML 的 AutoML、模型版本注册、
基于 SHAP 的局部预测解释、Experta 专家规则以及基于新标注数据的自适应再训练机制。
</p>

<p>
<a href="https://github.com/Maronari/AdaptiveML-DSS"><img src="https://img.shields.io/badge/GitHub-AdaptiveML--DSS-lightgrey?style=flat-square&logo=github" alt="GitHub"></a>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License">
</p>

<p style="font-size: 1.1em;">
<a href="README.md"><b>Русская локализация</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="README_en.md"><b>English Localization</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="README_ch.md"><b>中文本地化</b></a>
</p>
</div>

<p align="center">
<b>A. E. Дзгоев, D. S. Колганов, I. V. Корсунов, I. A. Филонов</b><br>
<i>莫斯科国立技术大学（МИРЭА）</i><br>
<i>俄罗斯，莫斯科，韦尔纳茨基大街 78 号，119454</i>
</p>

<h2 align="center">科学合作</h2>

<p>
AdaptiveML DSS 可作为自动化机器学习、预测解释和智能决策支持研究及实际应用的平台，
适用于能源、工业、数据分析以及其他领域。
</p>

<h2 align="center">引用</h2>

A. E. Дзгоев, D. S. Колганов, I. V. Корсунов, I. A. Филонов.
<i>面向复杂系统的自动化机器学习与决策支持平台：基于新实际数据的自适应回归再训练。</i>

<h2 align="center">摘要</h2>

现代表格数据自动化机器学习系统在新数据自适应再训练、局部预测解释和决策支持方面存在
不足。AdaptiveML DSS 将基于 LightAutoML 的 AutoML、数据集与模型版本管理、基于 SHAP
的局部解释以及基于 Experta 的决策支持系统整合到一个平台中。

平台支持 CSV/Excel 数据上传与验证、表格数据自动预处理、模型训练与保存、预测时的数据
模式兼容性检查、结果可视化以及自适应再训练。核心流程为：
<b>预测 → SHAP 因素 → 结构化事实 → 专家规则 → 建议</b>。
规则通过 JSON 配置，无需修改程序代码。

实验使用 2019–2022 年真实电力消费数据。训练集 R² = 0.9967，测试集 R² = 0.9721。
再训练实验还表明，质量下降的 Latest 模型不会替换当前 Champion 模型。

<b>关键词：</b> 自动化机器学习、AutoML、表格数据、SHAP、决策支持系统、DSS、
LightAutoML、自适应再训练、模型注册、电力消费预测、Experta、Web 界面。

<h2 align="center">引言</h2>

机器学习方法广泛应用于表格数据预测和分析。数据分布随时间变化可能导致模型精度下降，
因此需要质量控制和定期再训练。另一个问题是“黑箱”效应：仅给出数值预测并不能说明
哪些因素影响了结果。

AdaptiveML DSS 通过模型版本管理、受控再训练、SHAP 局部解释和确定性的专家规则，
将自动建模与可解释决策支持结合起来。

<h2 align="center">1. 现有解决方案综述</h2>

 涉及商业 AutoML 平台 H2O Driverless AI、DataRobot、Google AutoML Tables，以及
开源框架 LightAutoML、AutoGluon 和 FLAML。

所提出平台进一步提供数据集和模型版本管理、自适应再训练、局部预测解释以及
基于规则的决策支持。

<h2 align="center">2. 平台架构</h2>

平台包含五个主要功能组件：

1. 数据上传与验证模块；
2. AutoML 核心；
3. 模型与数据集注册中心；
4. 预测与解释服务；
5. 决策支持系统。

技术实现包括 FastAPI、`DatasetService`、`TrainingService`、`PredictionService`、
`ExplanationService`、`DecisionService`、SQLite、HTML/CSS/JavaScript Web 界面、
Chart.js 以及 Docker。

### 2.1 数据上传与验证

支持 CSV 和 Excel。系统识别数值、类别和日期/时间列，检查缺失值与重复值，
处理时间列并提取日历特征，同时在预测时检查输入数据模式兼容性。

### 2.2 AutoML 核心

LightAutoML 执行预处理、类别编码、算法选择以及基于 hold-out 数据的最佳流水线选择。
候选算法包括 LightGBM、CatBoost、XGBoost、RandomForest、线性模型和 MLP。

### 2.3 模型注册

每个模型拥有唯一 ID、版本和状态：

| 状态 | 含义 |
|---|---|
| **Champion** | 按主要质量指标选择的最佳模型，默认用于预测 |
| **Candidate** | 使用新数据训练、等待质量评估的候选模型 |
| **Latest** | 最新训练版本 |
| **Archived** | 保存用于历史记录或回退的旧版本 |

### 2.4 数据流

```text
用户 / 决策者
      |
      v
Web 界面
      |
      v
FastAPI API
      |
      +--------------------+
      |                    |
      v                    v
DatasetService       TrainingService
      |                    |
      v                    v
D1 数据集注册        D2 模型注册
                           |
                           v
                     D3 模型工件存储
                           |
                           v
                  预测 / 解释服务
                           |
                      +----+----+
                      |         |
                     SHAP      预测
                      |         |
                      +----+----+
                           |
                           v
                    DecisionService
                           |
                           v
                       D4 规则库
                           |
                           v
                     DSS 决策建议
```

<h2 align="center">3. 预测、解释与决策支持</h2>

预测服务加载当前 Champion 模型，检查输入数据模式并生成预测。

SHAP 用于计算局部特征贡献，界面显示影响最大的 3–5 个因素及其数值影响。

DSS 将预测值和 SHAP 因素转换为结构化事实，再通过 Experta 规则生成建议。
 包含 `manual_review` 和 `targeted_diagnostics` 等场景。规则集和触发条件可通过
JSON 修改，无需修改源代码。

<h2 align="center">4. 可视化与 Web 界面</h2>

界面支持项目管理、数据上传、训练、版本历史、预测和结果分析。交互式图表支持缩放、
预测区间选择、预测值、可用时的实际值、预测/误差表、SHAP 因素、DSS 建议以及 R²、
RMSE、AIC、BIC 等质量指标。

<h2 align="center">5. 自适应再训练</h2>

当新的标注实际数据到达时，可以训练新的模型版本。 支持用户触发再训练以及每日
训练候选模型。

新模型不会因为“最新”就自动替换 Champion。

| Model ID | 状态 | RMSE | 变化 |
|---|---|---:|---:|
| `model-cf9eb8376072` | Archived | 3.1248 | 初始模型 |
| `model-3e4cc357efc5` | Candidate | 2.7803 | −11.0% |
| `model-a36f46663244` | **Champion** | **2.7106** | −2.5% |
| `model-1d6878955cb0` | Latest | 2.9485 | +8.8% |

RMSE 从 3.1248 降至 2.7106，相对下降 13.2%。RMSE = 2.9485 的 Latest 模型因质量
下降而没有成为 Champion。

<h2 align="center">6. 实验评价</h2>

实验使用 2019–2022 年真实电力消费数据。 指出数据来源为俄罗斯 PJSC “Rosseti
North Caucasus”（JSC “Sevkavkazenergo”）。

评价指标包括 RMSE、MAE、R²、AIC 和 BIC。 还考察 24、72 和 168 小时预测范围，
并使用 MAE、RMSE、MAPE 和 R² 评价结果。

| 数据集 | R² |
|---|---:|
| Training | **0.9967** |
| Test | **0.9721** |

DSS 在预测之外提供 SHAP 解释和基于规则的结构化建议。

<h2 align="center">7. 实际应用</h2>

AdaptiveML DSS 可用于电力消费预测、表格数据分析、工业预测系统、模型质量监控、
智能决策支持以及可以用 JSON 形式表达专家规则的领域。

<h2 align="center">8. 结论</h2>

 提供了一个模块化平台，将自动回归建模、版本管理、预测、SHAP 解释、
Experta 决策支持和受控自适应再训练结合起来。

质量控制机制防止质量下降的新模型替换正在使用的 Champion。在真实电力消费数据上，
训练集 R² 为 0.9967，测试集 R² 为 0.9721。

<h2 align="center">9. 后续发展方向</h2>

- 使用 Dask 和 Apache Spark 处理超过 100 万行的数据集；
- 使用 Evidently 自动检测数据漂移/概念漂移；
- 开发 drag-and-drop 图形化规则编辑器；
- 构建基于 P10/P90 分位数的预测区间；
- 集成 MLflow 和 Kubeflow，实现持续部署与模型监控。

<h2 align="center">参考文献</h2>

1. Kustitskaya T.A., Esin R.V., Kytmanov A.A., Zykova T.V. Monitoring data and concept drift in industrial machine learning systems. 2026.
2. Sobolevsky V.A. Automation of machine learning model creation for time-series forecasting tasks. 2024. DOI: 10.17586/0021-3454-2024-67-11-951-957.
3. Kumaritov A.M., Dzgoev A.E., Babochiev O.R., Khuzmiev I.M. Modeling of a cost-control system for decision support in managing a power-grid enterprise. 2015.
4. Biryukov D.N., Suprun A.F. From the “black box” to transparency: foundations of explainability and interpretability in AI. 2025. DOI: 10.48612/jisp/x8ve-86ez-fv94.
5. Ageev V.A., Kazakov D.V., Repyev D.S. Review of traditional and neural-network methods for electric-load forecasting. 2023.
6. AdaptiveML DSS. GitHub repository. https://github.com/Maronari/AdaptiveML-DSS
7. H2O.ai. H2O Driverless AI.
8. DataRobot. Enterprise AI Platform.
9. Google Cloud. AutoML Tables.
10. LightAutoML. GitHub repository.
11. AutoGluon. AutoML for Tabular, Text, Image, and Time Series Data.
12. FLAML. A Fast and Lightweight AutoML Library.
13. Alkatsov M.I., Dzgoev A.E., Betrozov M.S. Research and development of a method for forecasting electricity consumption. 2012.
14. Khusnutdinov A.O., Khabarov V.I., Karmanov V.S. Deep learning for multivariate time-series analysis. 2025. DOI: 10.17212/2782-2001-2025-3-113-136.
15. Abotaleb M.S.A. Time-series forecasting algorithms based on weighted least absolute deviations. 2024.

<h2 align="center">附加信息</h2>

### 作者
A. E. Дзгоев、D. S. Колганов、I. V. Корсунов、I. A. Филонов，МИРЭА — Российский технологический университет。

<h2 align="center">许可证</h2>

项目采用 <b>Apache License 2.0</b>。
