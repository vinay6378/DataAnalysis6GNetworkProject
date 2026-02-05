# AI-Based Manufacturing Efficiency Classification Using Sensor, Production, and 6G Network Data

**Author:** Vinay Sharma (Data Science Intern, Unified Mentor)  
**Dataset:** `Thales_Group_Manufacturing.csv`  
**Target:** `Efficiency_Status` (High / Medium / Low)

## Abstract
Smart factories equipped with Industrial IoT sensors and high-speed 6G connectivity generate continuous streams of machine, production, quality, and network telemetry. Manufacturing efficiency can degrade quickly due to sensor instability, network latency/packet loss, or quality and error variations. This work proposes a supervised multi-class classification approach to predict the real-time manufacturing efficiency status (High/Medium/Low) using combined sensor, production, quality, and network signals. The solution follows a structured data science methodology comprising time-aware preprocessing, feature engineering, baseline and advanced model development, and explainability to build trust for engineers and operators.

## 1. Background and Problem Statement
Traditional dashboards are primarily descriptive (what happened). However, smart manufacturing requires near real-time, interpretable classification of current operational efficiency.

**Key challenges addressed:**
- Delayed detection of efficiency degradation
- Manual interpretation of many sensor + network metrics
- Lack of automated, interpretable efficiency assessment

**Objective:**
Predict `Efficiency_Status` (High/Medium/Low) at each timestamp using machine telemetry, quality/error indicators, and 6G network metrics.

## 2. Dataset Description
The dataset contains time-indexed observations at minute-level granularity.

**Columns (as provided):**
- `Date`, `Timestamp` (or `Time`)
- `Machine_ID`
- `Operation_Mode`
- `Temperature_C`, `Vibration_Hz`, `Power_Consumption_kW`
- `Network_Latency_ms`, `Packet_Loss_%`
- `Quality_Control_Defect_Rate_%`
- `Production_Speed_units_per_hr`
- `Predictive_Maintenance_Score`
- `Error_Rate_%`
- `Efficiency_Status` (target)

## 3. Data Science Methodology (Step-by-Step)
This project is implemented exactly as required in `Project_Detail.md`.

### 3.1 Data Preprocessing
**(a) Time-based ordering (Date + Time)**
- Combine `Date` + `Timestamp` into a single datetime `_dt`.
- Sort all rows by `_dt` to preserve temporal sequence.

**(b) Encode categorical variables (Operation_Mode)**
- Apply one-hot encoding to `Operation_Mode`.

**(c) Scale numerical features**
- Standardize numerical features using `StandardScaler`.

**(d) Address class imbalance (if any)**
- Compute class distribution of `Efficiency_Status`.
- If imbalance is significant, apply `class_weight="balanced"` for supported models.

### 3.2 Feature Engineering
**(a) Sensor stability indicators**
- Rolling standard deviation over a time window per `Machine_ID`:
  - `Temperature_C_roll_std_15`
  - `Vibration_Hz_roll_std_15`
  - `Power_Consumption_kW_roll_std_15`

**(b) Energy efficiency ratios**
- `Units_per_kW = Production_Speed_units_per_hr / Power_Consumption_kW`

**(c) Error-to-output ratios**
- `ErrorRate_per_Unit = Error_Rate_% / Production_Speed_units_per_hr`
- `DefectRate_per_Unit = Quality_Control_Defect_Rate_% / Production_Speed_units_per_hr`

**(d) Network reliability score**
- Construct a single reliability proxy from latency and packet loss:
  - Normalize latency and packet loss (robust quantile-based scaling)
  - `Network_Reliability_Score = 1 - (w_latency * latency_norm + w_loss * loss_norm)`

### 3.3 Model Development
**Baseline model**
- Multiclass Logistic Regression

**Advanced models**
- Random Forest Classifier
- Gradient Boosting / XGBoost (if enabled)

**Model comparison criteria**
- **Accuracy** (mean CV accuracy)
- **Stability** (std CV accuracy)
- **Interpretability** (linear coefficients / feature importance)

**Time-aware evaluation**
- Use `TimeSeriesSplit` cross-validation to avoid leakage across time.

## 4. Exploratory Data Analysis (EDA)
> Note: Insert plots/tables based on your executed analysis. This section is structured so you can directly paste outputs from notebooks or scripts.

### 4.1 Data quality checks
- Missing values per column
- Invalid timestamps or parsing failures
- Outliers (e.g., extreme latency, vibration spikes)

### 4.2 Target distribution (class imbalance)
- Class distribution (counts):
  - High: 2,986
  - Medium: 19,189
  - Low: 77,825
- Imbalance ratio (max/min): 26.0633

### 4.3 Time-series behavior
- Efficiency status over time (overall)
- Per-machine efficiency trends
- Efficiency by operation mode

### 4.4 Feature relationships
- Correlation heatmap (numerical features)
- Boxplots of key features by `Efficiency_Status` (e.g., latency, defect rate)

## 5. Experimental Results
After running training (`train_model.py`), summarize the contents of `metrics.json`.

### 5.1 Model comparison
Time-aware evaluation used `TimeSeriesSplit` (5 splits). Because the class imbalance ratio is 26.0633, training enabled `class_weight="balanced"` for models that support it.

| Model | CV Accuracy (Mean) | CV Accuracy (Std) | Notes |
|---|---:|---:|---|
| Logistic Regression | 0.9062 | 0.0019 | Baseline, interpretable |
| Random Forest | 0.9997 | 0.0003 | Best overall; stable |
| XGBoost | 0.9985 | 0.0005 | Strong performance |

### 5.2 Confusion matrix and per-class metrics
Below is the **last-fold** report for the best model (**Random Forest**). Label order for the confusion matrix is: **[High, Low, Medium]**.

**Confusion matrix (last fold):**

| True \ Pred | High | Low | Medium |
|---|---:|---:|---:|
| High | 519 | 0 | 0 |
| Low | 0 | 13029 | 0 |
| Medium | 0 | 0 | 3118 |

**Per-class metrics (last fold):**
- High: precision 1.000, recall 1.000, F1 1.000 (support 519)
- Low: precision 1.000, recall 1.000, F1 1.000 (support 13029)
- Medium: precision 1.000, recall 1.000, F1 1.000 (support 3118)

## 6. Feature Importance & Explainability
**Goal:** Identify top drivers of efficiency status and explain classifications to support trust.

### 6.1 Global explainability
- For tree-based models: plot top-20 feature importances.
- Report the most influential factors (examples):
  - Network instability (latency/packet loss)
  - Defect/error rates
  - Units per kW (energy efficiency)
  - Sensor stability (rolling std)

### 6.2 Local explainability (recommended)
- For a selected sample/timepoint, explain why prediction is Low/Medium/High.
- Recommended method: SHAP (TreeExplainer for RF/XGB, KernelExplainer for LR if needed).

## 7. Deployment Logic (Real-Time Readiness)
**Output per row:**
- Predicted efficiency label (High/Medium/Low)
- Confidence score (max predicted probability)

**Integration concept:**
- Stream incoming telemetry
- Apply preprocessing + feature engineering
- Generate prediction + confidence
- Trigger alerts when predicted efficiency is Low or when confidence is high and trend degrades

## 8. Streamlit Dashboard Implementation
A comprehensive interactive dashboard has been developed using Streamlit to provide real-time manufacturing efficiency analytics and explainability.

### 8.1 Dashboard Architecture
- **Framework:** Streamlit (Python web app framework)
- **Model Loading:** Loads the trained pipeline (`model.joblib`) and metrics (`metrics.json`)
- **Data Processing:** Applies identical preprocessing and feature engineering as training
- **Real-time Inference:** Processes user inputs and generates predictions with confidence scores

### 8.2 Key Features and Modules

#### 8.2.1 Efficiency Prediction Interface
- **Input Controls:** Sliders for all sensor, production, and network parameters
- **Live Classification:** Real-time prediction of Efficiency_Status (High/Medium/Low)
- **Confidence Visualization:** Gauge chart showing prediction confidence (0-100%)
- **Status Indicators:** Color-coded efficiency status with appropriate styling

#### 8.2.2 Machine Insights Panel
- **Machine Selection:** Dropdown to filter by specific Machine_ID
- **Operation Mode Filter:** Filter by Operation_Mode (Auto, Manual, Maintenance)
- **Historical Trends:** Visualizations of efficiency patterns over time
- **Performance Metrics:** Key operational indicators per machine

#### 8.2.3 Explainability Module
- **Global Feature Importance:** Bar chart showing top 20 influential features
- **Local SHAP Explanations:** Force plot explaining individual predictions
- **Feature Contribution Analysis:** Detailed breakdown of why a specific prediction was made
- **Decision Support:** Clear explanations for operators and engineers

#### 8.2.4 Operational Monitoring
- **Network Performance:** Real-time network reliability indicators
- **Quality Metrics:** Defect rates and error tracking
- **Energy Efficiency:** Units per kW consumption analysis
- **Alert Thresholds:** Configurable warning levels for key metrics

#### 8.2.5 User Interaction Features
- **Dynamic Filters:** Machine ID, operation mode, time range selection
- **Metric Sliders:** Interactive adjustment of all input parameters
- **Export Capabilities:** Download results and explanations
- **Responsive Design:** Optimized for desktop and tablet viewing

### 8.3 Technical Implementation Details
- **Feature Engineering Consistency:** Identical `add_feature_engineering` function ensures training/inference alignment
- **Model Pipeline Integration:** End-to-end preprocessing, scaling, and prediction
- **SHAP Integration:** TreeExplainer for local explainability of tree-based models
- **Performance Optimization:** Efficient data loading and caching for smooth user experience

### 8.4 Dashboard Benefits
- **Real-time Decision Support:** Immediate efficiency classification with confidence
- **Operator Trust:** Clear explanations build confidence in AI recommendations
- **Proactive Maintenance:** Early identification of efficiency degradation patterns
- **Process Optimization:** Data-driven insights for operational improvements

## 9. Key Performance Indicators (KPIs)
- **Efficiency Class:** High / Medium / Low
- **Prediction Confidence:** model certainty per prediction
- **Feature Contribution:** top drivers of predicted class (global + local)
- **Machine Efficiency Profile:** per-machine trends across time windows

## 10. Recommendations
**Operational recommendations (examples):**
- If efficiency degrades and `Network_Reliability_Score` drops:
  - Investigate 6G link quality, edge routing, congestion
- If defect/error ratios spike:
  - Inspect process parameters, tooling wear, calibration
- If sensor instability increases:
  - Check sensor mounting, maintenance schedule, environment conditions
- If energy efficiency (`Units_per_kW`) declines:
  - Evaluate load conditions, motor efficiency, and preventive maintenance

## 11. Limitations and Future Work
- Include calibration of predicted probabilities
- Add drift detection (concept drift across weeks/months)
- Perform hyperparameter tuning and thresholding for alert policies
- Add per-machine models or hierarchical modeling if machines differ significantly

## 12. Reproducibility (How to Run)
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Train and generate artifacts:
   - `python train_model.py --csv Thales_Group_Manufacturing.csv --out_model model.joblib --out_metrics metrics.json`
3. Launch dashboard:
   - `python -m streamlit run app.py` (or `streamlit run app.py` if streamlit is on PATH)

## Appendix: Files
- `train_model.py` (training + feature engineering + evaluation + artifacts)
- `app.py` (Streamlit dashboard)
- `metrics.json` (results summary)
- `model.joblib` (trained pipeline)

You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
  Network URL: http://10.194.209.73:8502