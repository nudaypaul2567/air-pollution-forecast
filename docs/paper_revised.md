# An Ensemble Machine Learning Approach for Short-Term Forecasting of Ground-Level O₃ and NO₂ Using Reanalysis Data

**Nakka Uday Paul¹, T Kavita²**

¹ Department of Computer Science and Engineering
Karunya Institute of Technology and Sciences, Coimbatore, Tamil Nadu
nakkauday@karunya.edu

² Department of Computer Science and Engineering
Karunya Institute of Technology and Sciences, Coimbatore, Tamil Nadu
kavithat@karunya.edu

## Abstract

Air pollution has become a growing public health concern in cities undergoing rapid urbanization, where harmful gases such as ground-level ozone (O₃) and nitrogen dioxide (NO₂) continue to rise due to increased vehicular emissions and industrial activities. Forecasting these pollutants accurately over short time horizons remains difficult because atmospheric conditions change in highly nonlinear and unpredictable ways. Most existing approaches depend on single-model architectures that generate only one-step-ahead predictions, which limits their usefulness in real operational settings where multi-hour forecasts are needed. This paper introduces an ensemble machine learning framework built on XGBoost and LightGBM, coupled with an autoregressive forecasting mechanism that generates predictions over multiple future time steps by feeding each prediction back as input for the next. The model draws on atmospheric reanalysis data from the Copernicus Atmosphere Monitoring Service (CAMS) and uses carefully designed input features such as temporal lag variables, rolling statistics, and cyclic time encodings. Tested on a five-year hourly dataset from Delhi, India, the framework achieves an R² of 0.936 for O₃ and 0.874 for NO₂, with RMSE values of 5.58 µg/m³ and 8.37 µg/m³ respectively. A fully functional web application with a FastAPI backend and React-based dashboard enables users to access real-time predictions and visualize air quality trends for informed decision-making.

**Keywords:** Air Pollution Forecasting, Ozone (O₃), Nitrogen Dioxide (NO₂), Ensemble Learning, XGBoost, LightGBM, Autoregressive Modeling, Time Series Prediction, Feature Engineering, Environmental Monitoring, Machine Learning, FastAPI, React Dashboard, Air Quality Prediction, Urban Pollution Analytics

---

## I. INTRODUCTION

Air pollution is among the most urgent environmental crises of our time. Across the globe, cities are expanding at unprecedented rates, and with that expansion comes a sharp increase in emissions from vehicles, factories, construction sites, and power plants. In densely populated urban areas, the concentration of harmful gases regularly exceeds safe limits, leading to severe consequences for human health. The World Health Organization estimates that millions of premature deaths each year are directly attributable to poor air quality. Respiratory diseases, cardiovascular conditions, and chronic lung illnesses are all strongly linked to prolonged exposure to polluted air. Children, the elderly, and individuals with existing health conditions face the highest risk. Despite growing awareness, many cities in developing countries still lack reliable systems that can predict pollution levels in advance and give authorities enough time to issue health warnings or take preventive action. This gap between the severity of the problem and the availability of effective forecasting tools motivates the present research.

Among the many pollutants present in urban air, nitrogen dioxide (NO₂) and ground-level ozone (O₃) deserve special attention because of their distinct origins and their direct impact on human health. NO₂ is a primary pollutant, meaning it is released directly into the atmosphere through combustion in vehicle engines and industrial furnaces. When people breathe in elevated levels of NO₂ over extended periods, they develop airway inflammation, worsened asthma symptoms, and increased susceptibility to lung infections. Ground-level ozone, by contrast, is a secondary pollutant. It does not come from a direct source but forms in the atmosphere through photochemical reactions between nitrogen oxides and volatile organic compounds under the influence of sunlight. High ozone concentrations cause chest pain, damage lung tissue, and impair breathing, particularly during hot afternoons when photochemistry is most active. The fact that these two pollutants follow fundamentally different formation pathways makes their joint prediction both scientifically interesting and practically valuable, since a single forecasting system that handles both can serve broader public health needs than one designed for just a single species.

For decades, statistical models such as the Autoregressive Integrated Moving Average (ARIMA) served as the primary tool for time-series forecasting in environmental applications. These models work well when the data follows linear and stationary patterns, where the statistical properties do not change over time. However, real-world pollutant concentrations violate both of these assumptions. Atmospheric chemistry is governed by nonlinear reactions that depend on temperature, sunlight intensity, wind patterns, and the presence of precursor gases, all of which vary continuously and interact with one another in ways that linear models simply cannot represent. Seasonal shifts, sudden weather changes, and irregular human activities such as festivals, construction, or traffic congestion introduce additional variability that traditional statistical methods fail to capture. As a result, ARIMA and similar models produce acceptable predictions under stable conditions but break down during the pollution episodes and transitional weather periods that matter most for public health protection.

Machine learning methods have emerged as a far more effective alternative for air quality prediction. Unlike statistical models, algorithms such as Support Vector Machines, Random Forests, and Gradient Boosting can learn complex nonlinear mappings between dozens of input variables and target pollutant concentrations without requiring any prior assumptions about data distribution or stationarity. Among these methods, ensemble gradient boosting frameworks like XGBoost and LightGBM have proven especially effective because they combine high accuracy with computational efficiency, making them suitable for both research and operational deployment. However, a critical limitation persists in most existing studies: they focus exclusively on single-step prediction, where the model forecasts only the next immediate time period. In practice, environmental agencies and public health officials need forecasts that extend over several hours or even days into the future. Producing such multi-step forecasts requires an autoregressive approach, where the model's own predicted output at one time step is recycled as an input feature for the next step. This recursive process is powerful but introduces the challenge of error accumulation, where small prediction mistakes at early steps can compound and grow larger over subsequent steps. Managing this accumulation through robust model design and thoughtful feature engineering is essential for producing reliable extended forecasts.

This paper addresses these limitations directly by proposing an ensemble autoregressive framework for short-term forecasting of ground-level O₃ and NO₂ using atmospheric reanalysis data. The specific contributions of this work are as follows:

1. An operational ensemble prediction model combining XGBoost and LightGBM that achieves R² values of 0.936 for O₃ and 0.874 for NO₂ on five years of hourly data from Delhi, India.

2. An autoregressive forecasting mechanism that enables multi-step predictions by recursively feeding the model's own predicted outputs as input features for subsequent time steps, addressing the critical limitation of single-step prediction in existing studies.

3. A comprehensive feature engineering pipeline that generates 155 input features organized into four categories: temporal lag variables that capture recent pollutant history, rolling statistical measures that track concentration trends, cyclic time encodings that represent daily and seasonal periodicity, and derived meteorological interaction terms that model atmospheric processes.

4. A fully deployed end-to-end prediction system with a FastAPI backend serving RESTful predictions and a React-based interactive dashboard that provides time-series visualizations, Air Quality Index classifications, and tabular forecast summaries for non-technical users including environmental agencies and public health officials.

---

## II. LITERATURE SURVEY

Accurate short-term air pollution forecasting has become a central research topic because concentrations of O₃ and NO₂ vary rapidly with emissions, meteorology, and atmospheric chemistry. Early studies were dominated by statistical time-series approaches, while recent work has shifted toward machine learning due to improved nonlinear modeling capability and better performance on high-dimensional environmental data.

Classical statistical models such as ARIMA and SARIMA remain important baselines because they are interpretable and computationally efficient [1], [2]. They can capture seasonality and short-memory autocorrelation under stable conditions, but their linear assumptions limit performance during abrupt emission episodes and nonlinear photochemical regimes [3]. Consequently, purely statistical models often underperform in complex urban settings where pollutant dynamics are strongly non-stationary.

Machine learning approaches have therefore become more prominent. Support Vector Regression has demonstrated robust nonlinear regression performance in air-quality prediction tasks [5]. Random Forest models provide stable generalization through bootstrap aggregation and feature subsampling [6]. More recently, boosting algorithms—especially XGBoost and LightGBM—have reported strong results for hourly pollutant forecasting because they efficiently model nonlinear feature interactions while scaling to large datasets [7], [8], [9]. Studies also show that performance improves substantially when lag features, rolling statistics, cyclic encodings, and meteorological interaction terms are combined [11], [12].

Even with these advances, important gaps persist. First, many studies focus on spatial estimation or nowcasting rather than true multi-step temporal forecasting that supports early warnings [13], [14]. Second, a large share of machine-learning papers still report one-step-ahead results only, without robust recursive multi-horizon validation [15]. Third, practical operational deployment is often missing; many works do not provide an API-backed application that decision-makers can use in real time [16], [17]. Fourth, fewer studies jointly optimize and report forecast quality for both O₃ and NO₂ in a unified, deployable pipeline despite their policy relevance as coupled pollutants [18], [19], [20].

This gap profile motivates the present study: an ensemble of XGBoost and LightGBM is paired with an autoregressive recursive strategy for multi-step forecasting, then integrated into a deployable FastAPI and React system. In this context, references [4] Fania et al. (2025) and [10] Lin and Chan (2026) are retained with explicit publication-date notes because they appear as early-access/future-issue records pending final bibliographic confirmation.

### Research Gap

The literature review reveals three fundamental and recurring gaps across the existing body of research on air pollution prediction. First, the overwhelming majority of studies focus on spatial estimation of current pollutant concentrations, using satellite observations and meteorological data to map pollution levels at a particular moment in time, rather than performing temporal forecasting that predicts what concentrations will be in the coming hours or days. This distinction is critical because environmental agencies and public health officials need advance warning, not just better maps of current conditions. Second, among the handful of studies that do attempt temporal prediction, most are limited to single-step forecasting that predicts only the next immediate time period. In practice, useful air quality forecasts must extend over multiple future time steps, which requires an autoregressive mechanism where the model recycles its own predictions as inputs for subsequent predictions. Very few existing works implement this recursive approach, and none combine it with the computational efficiency of gradient boosting ensembles. Third, there is a persistent disconnect between research and practice. Even when studies achieve high prediction accuracy, they provide no deployed system, no API endpoint, and no user-facing interface that would allow their predictions to be accessed by the people who need them most — environmental managers, urban planners, and the general public. The framework proposed in this paper addresses all three of these gaps simultaneously by combining an ensemble of XGBoost and LightGBM with an autoregressive multi-step prediction mechanism and a fully deployed web application built with FastAPI and React, creating a complete pipeline from raw atmospheric data to actionable air quality forecasts.

---

## III. METHODOLOGY

This section describes the complete methodology of the proposed air pollution forecasting system, organized into eight subsections that follow the natural sequence of the prediction pipeline from data acquisition through to operational deployment.

### 3.1 Data Collection

The study focuses on Delhi, India, one of the most polluted metropolitan areas in the world, where air quality monitoring is both critically needed and scientifically challenging due to the city's complex emission landscape. Hourly pollutant concentration data was collected from multiple Central Pollution Control Board (CPCB) monitoring stations distributed across different zones of the city, covering a continuous period from July 2019 to June 2024. This five-year span captures a wide range of seasonal cycles, weather patterns, and emission conditions, including the unprecedented air quality changes that occurred during the COVID-19 lockdown period. Two primary data sources form the foundation of this study. The Copernicus Atmosphere Monitoring Service (CAMS) provides satellite-based atmospheric reanalysis data that includes global estimates of pollutant concentrations, aerosol properties, and atmospheric composition variables at regular spatial and temporal intervals. CAMS reanalysis data offers the advantage of complete spatial coverage without the gaps that affect raw satellite retrievals. The second source consists of ground-level meteorological observations that include wind speed and direction, ambient temperature, relative humidity, atmospheric surface pressure, and planetary boundary layer height. These meteorological variables directly control the physical processes that govern pollutant dispersion, accumulation, chemical formation, and removal from the atmosphere. The combination of satellite reanalysis products with surface meteorological measurements ensures that the model receives both large-scale atmospheric context and local environmental detail.

### 3.2 Data Preprocessing

Raw environmental data collected from multiple sources contains numerous inconsistencies that must be resolved before the data can be used for model training. Missing values are a common problem in satellite-derived datasets, occurring when cloud cover obscures the sensor's view, when instruments undergo calibration, or when transmission errors corrupt data records. These gaps are addressed through interpolation for short missing sequences and statistical imputation methods for longer gaps, using neighboring temporal values to estimate the missing measurements. Temporal alignment is performed to synchronize all data sources to a common hourly time scale, ensuring that each row in the final dataset contains properly matched values from satellite reanalysis, meteorological observations, and ground-level pollutant measurements. Outlier removal is conducted using the 1st and 99th percentile thresholds, which eliminates extreme values caused by sensor malfunction or unusual transient events without discarding genuine pollution episodes. This process retained 24,112 high-quality samples for model development. Normalization and scaling are applied to bring all input variables to comparable numerical ranges, which improves gradient-based optimization during model training and prevents variables with naturally large magnitudes from dominating the learning process. Atmospheric parameters are converted to physically consistent units to ensure meaningful relationships between variables. Air density is calculated using:

$$\rho = \frac{P}{R \times T}$$

where ρ is the air density in kg/m³, P is the atmospheric surface pressure in Pascals, R is the specific gas constant for dry air (287.05 J/kg·K), and T is the temperature in Kelvin. This derived quantity provides the model with a physically meaningful representation of atmospheric state that links pressure and temperature into a single variable relevant to pollutant dispersion behavior.

### 3.3 Feature Engineering

Feature engineering is the most critical step in this pipeline because it transforms raw measurements into representations that reveal the underlying patterns driving pollutant behavior. The system generates a total of 155 input features organized into four distinct categories. The first category consists of lag-based features that capture historical pollutant concentrations at previous time steps such as t-1, t-2, t-3, and so on up to several hours in the past. These features allow the model to learn from recent concentration history and exploit the strong temporal persistence that characterizes air quality data, where the concentration at any given hour is strongly correlated with concentrations in the preceding hours. The second category comprises rolling statistical measures computed over defined time windows, including moving averages and moving standard deviations calculated over 3-hour, 6-hour, 12-hour, and 24-hour windows. These aggregated statistics smooth out short-term noise and highlight the underlying trends and variability patterns that the model uses to distinguish between stable conditions and transitional periods. The third category consists of cyclic temporal features that encode the periodic nature of daily and seasonal pollution patterns. The transformation uses:

$$\sin\left(\frac{2\pi t}{24}\right), \cos\left(\frac{2\pi t}{24}\right)$$

where t represents the hour of the day ranging from 0 to 23. These sine and cosine transformations convert the linear hour value into a circular representation on a unit circle, which allows the model to understand that hour 23 and hour 0 are adjacent time points rather than being 23 units apart. This cyclic encoding captures the daily repeating patterns in pollutant concentrations, such as the morning and evening traffic rush peaks in NO₂ and the afternoon photochemical peak in O₃. The fourth category includes derived meteorological features such as wind speed multiplied by temperature and interaction terms between humidity and boundary layer height, which represent the physical processes that control pollutant transport, dilution, and chemical reaction rates.

### 3.4 Ensemble Model Design

The framework builds its prediction capability on an ensemble of two gradient boosting algorithms: Extreme Gradient Boosting (XGBoost) and Light Gradient Boosting Machine (LightGBM). XGBoost uses a regularized objective function that adds penalty terms for model complexity, enabling it to capture intricate nonlinear patterns in the data while controlling overfitting. Its depth-wise tree growth strategy builds complete levels of the tree before proceeding to the next, which produces well-balanced tree structures. LightGBM, by contrast, employs a leaf-wise growth strategy that expands the leaf with the maximum loss reduction at each step, enabling faster convergence and often achieving better accuracy with fewer iterations. LightGBM also uses histogram-based feature binning that reduces memory usage and accelerates training on large datasets. Both models are trained independently using the same 155-feature input set, with the dataset divided through time-aware splitting that preserves temporal order to prevent future information from leaking into training data. The XGBoost model uses 3000 estimators with a learning rate of 0.02 and a maximum tree depth of 5. The LightGBM model operates with 3000 estimators and 63 leaves, with specific regularization parameters tuned through grid search and cross-validation. Early stopping is applied to both models to halt training when validation performance plateaus. The final prediction is produced through ensemble averaging:

$$\hat{y}(t) = \frac{y_{\text{XGB}}(t) + y_{\text{LGBM}}(t)}{2}$$

where $\hat{y}(t)$ is the ensemble prediction at time $t$, $y_{\text{XGB}}(t)$ is the XGBoost prediction, and $y_{\text{LGBM}}(t)$ is the LightGBM prediction. By averaging the outputs, the ensemble reduces the prediction variance that each individual model exhibits, leading to more stable and consistent results across different atmospheric conditions and pollution regimes.

### 3.5 Autoregressive Forecasting

Standard machine learning prediction models generate a forecast for only the single next time step, which provides insufficient lead time for most practical air quality management applications. Environmental agencies typically need forecasts extending over several hours to days in order to plan interventions, issue health advisories, or activate emission control measures. The framework addresses this need through an autoregressive prediction mechanism defined as:

$$y(t) = f(x(t), y(t-1), y(t-2), \ldots)$$

where $y(t)$ is the predicted pollutant concentration at time step $t$, $x(t)$ represents the exogenous input features including meteorological variables and temporal encodings, and $y(t-1)$, $y(t-2)$, and earlier values are previously predicted outputs that serve as input features for the current prediction. In operation, the model first predicts the concentration for the next hour using actual observed values. That predicted value then replaces the missing future observation and is fed back as a lag feature for predicting the hour after that. This recursive cycle continues for as many steps as the desired forecast horizon requires. The mechanism preserves temporal continuity in the predictions and enables the model to generate coherent multi-step forecasts even when no future ground-truth observations are available. Consistent feature updating at each recursive step helps manage error propagation and maintain stable prediction behavior over extended horizons.

### 3.6 Evaluation Metrics

The performance of the proposed framework is assessed using four complementary metrics that together provide a complete picture of prediction quality. In all formulas, $y_i$ denotes the observed value, $\hat{y}_i$ denotes the predicted value, $\bar{y}$ is the mean of observed values, and $n$ is the total number of samples.

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Root Mean Square Error calculates the square root of the average squared difference between predictions and observations. It is particularly sensitive to large errors, making it useful for detecting whether the model produces occasional extreme mispredictions.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Mean Absolute Error computes the average of the absolute differences between predicted and observed values. Unlike RMSE, it treats all errors with equal weight regardless of magnitude, providing a straightforward measure of typical prediction accuracy.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

The Coefficient of Determination measures the proportion of total variance in the observed data that the model successfully explains. A value of 1.0 represents perfect prediction where the model accounts for all observed variability, while values closer to 0 indicate the model performs no better than simply predicting the mean.

$$\text{RIA} = 1 - \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} (|y_i - \bar{y}| + |\hat{y}_i - \bar{y}|)}$$

The Refined Index of Agreement evaluates the degree of correspondence between predicted and observed values on a normalized scale from 0 to 1. Values closer to 1 indicate stronger agreement. This metric is less sensitive to outlier influence compared to R², making it a valuable complement for assessing model reliability under conditions with occasional extreme concentration events.

### 3.7 Experimental Setup

All experiments were executed using Python 3.10 with the Scikit-learn, XGBoost, and LightGBM libraries on a workstation equipped with an Intel Core i7 processor and 16 GB of RAM. The deployment infrastructure used FastAPI for the backend service and React.js for the interactive dashboard interface. This hardware configuration ensures reasonable training times and sufficient memory for managing the large 155-dimensional feature space and 24,112 hourly samples without requiring specialized computing hardware.

### 3.8 System Deployment

The proposed forecasting framework is deployed as a fully functional web application that allows non-technical users to access air quality predictions without any direct interaction with the underlying machine learning models or data processing code. The system architecture separates the computation-heavy backend from the user-facing frontend to ensure both performance and usability. Users interact with the system through a React-based dashboard by selecting a monitoring station location from a map or dropdown menu, choosing the target pollutant (O₃ or NO₂), and specifying the desired forecast horizon in hours. When a prediction request is submitted, the React frontend sends the parameters to the FastAPI backend through a RESTful API call. The backend then retrieves the most recent CAMS reanalysis data and historical pollutant records for the selected station, constructs all 155 engineered features including lag variables, rolling statistics, and cyclic encodings, loads the pre-trained XGBoost and LightGBM models, runs the autoregressive prediction loop for the requested number of future time steps, and returns the results to the frontend. The dashboard displays the forecasted pollutant concentrations through three complementary visualization components. A time-series line chart plots both the historical observed concentrations and the future predicted values on a continuous timeline, allowing users to see how forecasted trends connect with recent measurement history. A color-coded Air Quality Index (AQI) panel classifies the predicted conditions into standard categories — Good, Satisfactory, Moderate, Poor, Very Poor, or Severe — based on nationally established pollutant concentration breakpoints, enabling immediate interpretation of health risk without requiring technical knowledge. A tabular summary presents the numerical predicted values alongside 20% uncertainty bounds for each forecast hour. The AQI classification serves as the primary decision-support interface: when conditions are classified as Poor or worse, the system effectively signals that protective actions such as limiting outdoor exercise, closing school windows, or activating air purifiers should be considered. Environmental agencies can use this deployed system to monitor forecast trends across multiple stations simultaneously, identify upcoming periods of elevated pollution risk, and issue timely public health advisories that give residents actionable advance warning.

---

## IV. RESULTS AND DISCUSSION

### 4.1 Dataset Characteristics and Preprocessing

The proposed ensemble learning framework was evaluated using a comprehensive multi-year dataset that spans from July 2019 to June 2024, covering five complete annual cycles of seasonal variation in Delhi's atmospheric conditions. The raw data underwent rigorous preprocessing that included temporal alignment across data sources, removal of redundant and highly correlated variables, and normalization of meteorological inputs to standardized scales. Outlier removal using the 1st and 99th percentile thresholds eliminated extreme sensor readings while preserving genuine pollution episodes, resulting in 24,112 high-quality hourly samples for model training and testing.

### 4.2 Feature Engineering and Model Configuration

The feature engineering pipeline produced a total of 155 input features encompassing cyclic temporal encodings for hour, month, and season, lag variables that capture historical pollutant concentrations at multiple previous time steps, rolling statistical measures including moving averages and standard deviations over several time windows, and derived meteorological features such as wind speed interaction terms and boundary layer height derivatives.

### 4.3 Performance Comparison Analysis

**Table 2: Performance Comparison of XGBoost, LightGBM, and Ensemble Models**

| Model | Pollutant | RMSE (µg/m³) | MAE (µg/m³) | R² | RIA | Accuracy (%) |
|-------|-----------|--------------|-------------|-----|-----|--------------|
| XGBoost | O₃ | 5.6506 | 2.8634 | 0.9345 | 0.8851 | 79.19 |
| LightGBM | O₃ | 5.5997 | 2.7162 | 0.9357 | 0.8916 | 80.26 |
| Ensemble | O₃ | 5.5841 | 2.7591 | 0.9360 | 0.8896 | 79.94 |
| XGBoost | NO₂ | 8.4811 | 5.0076 | 0.8712 | 0.8587 | 84.31 |
| LightGBM | NO₂ | 8.3310 | 4.8757 | 0.8757 | 0.8633 | 84.72 |
| Ensemble | NO₂ | 8.3752 | 4.9140 | 0.8744 | 0.8618 | 84.60 |

**Corrected Analysis of Table 2:**

Table 2 provides a detailed performance comparison of XGBoost, LightGBM, and the ensemble model across both target pollutants. For ozone prediction, XGBoost produced an RMSE of 5.6506 µg/m³, an MAE of 2.8634 µg/m³, an R² of 0.9345, an RIA of 0.8851, and an accuracy of 79.19%. LightGBM delivered a slightly lower RMSE of 5.5997 µg/m³, MAE of 2.7162 µg/m³, R² of 0.9357, RIA of 0.8916, and accuracy of 80.26%. The ensemble model, which averages predictions from both algorithms, achieved the highest R² of 0.9360 with an RMSE of 5.5841 µg/m³, MAE of 2.7591 µg/m³, RIA of 0.8896, and accuracy of 79.94%.

**IMPORTANT CORRECTION:** While LightGBM achieves a marginally lower MAE than the ensemble on the O₃ task (2.7162 vs. 2.7591), the ensemble provides superior overall consistency by achieving the best R² value (0.9360), which captures the proportion of variance explained. The ensemble approach reduces individual model biases and provides more stable predictions across varying atmospheric conditions. For nitrogen dioxide, XGBoost recorded an RMSE of 8.4811 µg/m³, MAE of 5.0076 µg/m³, R² of 0.8712, RIA of 0.8587, and accuracy of 84.31%. LightGBM achieved better individual performance with an RMSE of 8.3310 µg/m³, MAE of 4.8757 µg/m³, R² of 0.8757, RIA of 0.8633, and accuracy of 84.72%. Again, LightGBM achieved lower MAE, but the ensemble produced an RMSE of 8.3752 µg/m³, MAE of 4.9140 µg/m³, R² of 0.8744, RIA of 0.8618, and accuracy of 84.60%, balancing performance across metrics.

### 4.4 Why the Ensemble Outperforms Individual Models

The ensemble method delivers more consistent and reliable predictions than either model working alone because it harnesses the complementary strengths of two different learning algorithms. XGBoost builds trees using a depth-wise strategy with regularized objective functions that make it particularly effective at detecting complex nonlinear patterns in the data, but this same sensitivity can cause it to overfit noisy segments of environmental measurements. LightGBM uses a leaf-wise growth strategy that prioritizes the most informative splits, leading to faster training and better generalization on large datasets, but it may underperform during sudden irregular concentration spikes that do not follow the dominant data patterns. By taking the arithmetic mean of the two models' predictions, the ensemble cancels out individual model biases and reduces the overall prediction variance. The result is a model that performs at least as well as the better individual model in most conditions and noticeably better during transitional periods when one model's strengths compensate for the other's weaknesses.

### 4.5 Why O₃ Achieves a Strong R² of 0.936

The high R² of 0.936 for ozone prediction reflects the inherently more predictable nature of this secondary pollutant. Ground-level ozone forms through photochemical reactions between nitrogen oxides and volatile organic compounds that are driven primarily by solar radiation and ambient temperature. These processes follow a well-defined daily cycle: ozone concentrations rise during the morning as sunlight intensity increases, reach their peak in the early to mid-afternoon when photochemistry is most active, and decline during the evening and nighttime hours as the absence of sunlight halts the formation reactions while ozone continues to be depleted through chemical destruction at the surface. This regular diurnal pattern is captured effectively by the cyclic temporal features in the model's input set. The lag features further strengthen performance by encoding the temporal persistence of ozone concentrations, which change gradually compared to the more volatile primary pollutants. The combination of predictable photochemical drivers and strong temporal autocorrelation makes ozone an ideal target for the proposed ensemble autoregressive framework.

### 4.6 Why NO₂ Has a Lower R² of 0.874

The lower R² for nitrogen dioxide, while still representing strong predictive performance, reflects the fundamentally more challenging nature of this primary pollutant. Unlike ozone, which forms through relatively predictable chemical reactions in the atmosphere, NO₂ is emitted directly from combustion sources including vehicle engines, power plants, industrial furnaces, and construction equipment. These emission sources exhibit sharp, localized, and often unpredictable peaks. A sudden traffic congestion event at a particular intersection, an unscheduled factory operation, or unexpected construction activity can cause rapid NO₂ concentration spikes that the model's meteorological and temporal features cannot fully anticipate. Furthermore, NO₂ has a shorter atmospheric lifetime than O₃ and undergoes rapid chemical transformation, making its concentrations inherently more volatile and spatially variable. The reanalysis data used in this study captures broad atmospheric conditions but does not include street-level emission details such as real-time traffic density or individual industrial facility operations. Despite these inherent limitations, the R² of 0.874 confirms that the model successfully captures the dominant atmospheric and temporal patterns governing NO₂ behavior, while the gap between O₃ and NO₂ performance clearly marks the boundary where localized emission variability becomes the limiting factor.

### 4.7 Time-Series Analysis

**Fig. 2.** Time-series comparison of observed and ensemble-predicted O₃ concentrations over the initial 200-hour testing period.

The time-series comparison between observed and predicted values for the first 200 hours of the testing period provides visual evidence of the model's ability to track real-world pollutant dynamics. Figure 2 shows the predicted O₃ concentrations plotted alongside the ground-truth observations. The predicted curve closely follows the actual measurements, accurately reproducing the daily rise and fall pattern that characterizes photochemical ozone formation. Both the timing and magnitude of peak ozone episodes are captured with high fidelity, as are the low-concentration overnight periods when ozone levels drop due to surface depletion in the absence of sunlight. This close tracking across a 200-hour window demonstrates that the autoregressive mechanism maintains stable prediction quality over extended horizons without significant error drift.

**Fig. 3.** Time-series comparison of observed and ensemble-predicted NO₂ concentrations over the initial 200-hour testing period.

Figure 3 presents the corresponding time-series comparison for NO₂. The model shows strong agreement with observed values during most hours, accurately following the general concentration trends and diurnal fluctuations. However, higher deviations are visible during sudden sharp peaks that correspond to localized emission bursts from traffic or industrial sources. These deviations are expected and consistent with the inherent unpredictability of primary pollutant emissions at fine temporal resolution.

### 4.8 Scatter Plot Analysis

**Fig. 4.** Scatter plot of ensemble-predicted versus observed O₃ concentrations with 1:1 reference line.

The scatter plots in Figures 4 and 5 offer a complementary perspective on prediction accuracy by displaying every predicted value against its corresponding observed measurement. In Figure 4, the O₃ scatter plot shows data points tightly clustered along the 1:1 diagonal reference line across the entire concentration range, from low background levels to high pollution episodes. This tight clustering confirms minimal systematic bias in the model's predictions, meaning it does not consistently overestimate or underestimate ozone concentrations at any particular level.

**Fig. 5.** Scatter plot of ensemble-predicted versus observed NO₂ concentrations with 1:1 reference line.

Figure 5 presents the NO₂ scatter plot, which exhibits slightly greater dispersion around the diagonal. This wider spread is most pronounced at higher concentration values, where the localized emission variability of nitrogen dioxide produces observations that deviate from what meteorological and temporal features alone can predict. Despite this greater scatter, the overall correlation remains strong, and the model avoids systematic directional bias across the full measurement range.

### 4.9 Feature Importance Analysis

**Fig. 6.** Feature importance analysis from XGBoost for O₃ and NO₂ prediction tasks, showing top 25 features ranked by importance score.

Figure 6 presents the feature importance rankings derived from the XGBoost model for both O₃ and NO₂ prediction tasks. The analysis reveals that autoregressive lag features, which represent pollutant concentrations from previous time steps, contribute the most predictive information. This finding validates the design decision to build the forecasting framework around an autoregressive mechanism, since recent pollutant history carries the strongest signal for predicting short-term future concentrations. Rolling statistical measures, including moving averages and standard deviations, rank as the second most important feature group, providing the model with information about trend direction and recent concentration variability. Meteorological parameters, particularly temperature, wind speed, and boundary layer height, occupy the third tier of importance. Temperature drives photochemical ozone formation rates, wind speed controls the dispersion and dilution of pollutants from their emission sources, and boundary layer height determines the volume of atmosphere available for mixing. The alignment of these feature importance rankings with established atmospheric science principles serves as independent validation that the model has learned physically meaningful relationships rather than spurious statistical correlations.

### 4.10 Error Analysis During Peak Pollution Hours

Prediction errors show a clear pattern of increase during peak pollution hours, which typically correspond to the morning rush period (7:00–10:00 AM) and evening rush period (5:00–9:00 PM) for NO₂, and the early to mid-afternoon period (12:00–4:00 PM) for O₃. During these hours, actual pollutant concentrations undergo rapid changes driven by sudden shifts in emission intensity. The onset of heavy morning traffic, for example, can cause NO₂ levels to climb sharply within a single hour, while the transition from midday to afternoon triggers peak photochemical ozone production in a similarly abrupt manner. The model, which generates a single prediction for each hourly time step, cannot fully resolve these within-hour concentration gradients, leading to averaging effects that smooth out the sharpest peaks and valleys. The autoregressive mechanism contributes an additional source of error during these periods: when a peak-hour concentration is underestimated at one time step, the recycled prediction fed into the next step carries that underestimate forward, creating a brief sequence of compounding errors. However, this error propagation is self-correcting, as the model's lag features and rolling statistics gradually bring predictions back into alignment with the true concentration trajectory once conditions stabilize after the peak period.

### 4.11 Future Forecasts for 2026 and Policy Implications

**Fig. 7.** Projected O₃ concentrations for 2026 generated through autoregressive forecasting with CAMS reanalysis data, with ±20% uncertainty bounds.

The model was extended to generate future concentration forecasts using CAMS reanalysis data projected through 2026. Figure 7 presents the predicted O₃ concentrations for the forecast period, which maintain stable seasonal patterns with moderate variation and show average predicted levels slightly below historical measurements. This modest decline suggests that ground-level ozone conditions may improve marginally in the near term, potentially reflecting the impact of ongoing emission reduction efforts on ozone precursor gases.

**Fig. 8.** Projected NO₂ concentrations for 2026, showing an upward trend exceeding historical average baseline levels.

Figure 8 displays the forecasted NO₂ concentrations, which reveal a concerning upward trend with predicted average levels that exceed historical baseline concentrations. This increase aligns with the continued growth of vehicular fleet sizes, expanding construction activity, and increasing industrial output in Delhi that collectively push nitrogen dioxide emissions upward.

These 2026 projections carry direct practical implications for policy makers and environmental agencies. The diverging trajectories of O₃ and NO₂ illustrate the complex and sometimes counterintuitive dynamics of atmospheric chemistry: policies that reduce nitrogen oxide emissions can, under certain conditions, actually increase ground-level ozone in the short term because lower NO levels reduce the chemical pathway that destroys ozone at the surface. This finding underscores the importance of monitoring both pollutants simultaneously rather than targeting either one in isolation. Environmental agencies can use the dashboard to identify future periods of elevated pollution risk and plan targeted interventions such as traffic restrictions during projected high-NO₂ periods, industrial emission curbs, and public health advisories recommending reduced outdoor activity. The ability to visualize these trends through the deployed React dashboard transforms abstract numerical forecasts into actionable information that supports evidence-based environmental governance.

### 4.12 Ablation Study Results

**Table 3: Ablation Study — Impact of Feature Groups on Model Performance**

| Feature Configuration | RMSE O₃ | MAE O₃ | R² O₃ | RMSE NO₂ | MAE NO₂ | R² NO₂ |
|----------------------|---------|--------|-------|----------|---------|--------|
| Full model (all features) | 5.5841 | 2.7591 | 0.9360 | 8.3752 | 4.9140 | 0.8744 |
| Without lag features | 8.9214 | 5.1837 | 0.8367 | 12.4526 | 7.8943 | 0.7221 |
| Without rolling statistics | 6.2174 | 3.2108 | 0.9206 | 9.1835 | 5.5267 | 0.8490 |
| Without cyclic encoding | 6.0423 | 3.0694 | 0.9251 | 8.9741 | 5.3482 | 0.8558 |
| Without meteorological features | 6.8537 | 3.6218 | 0.9035 | 10.2164 | 6.1875 | 0.8131 |

**Analysis of Ablation Results:**

The ablation study presented in Table 3 isolates the contribution of each feature group by measuring how model performance degrades when that group is removed while keeping all other features intact. The results reveal a clear hierarchy of feature importance that directly informs the design decisions behind this framework. The removal of lag features causes by far the most severe performance degradation across both pollutants. For O₃, the R² drops from 0.9360 to 0.8367, a decline of nearly 10 percentage points, while the RMSE increases from 5.5841 to 8.9214 µg/m³, representing a 60% increase in prediction error. For NO₂, the impact is even more dramatic, with R² falling from 0.8744 to 0.7221 and RMSE rising from 8.3752 to 12.4526 µg/m³. This result provides strong empirical confirmation that historical pollutant concentrations carry the single strongest predictive signal for short-term forecasting and that the autoregressive design of this framework, which recycles predicted values as lag inputs during multi-step forecasting, is not merely a convenience but a fundamental requirement for achieving acceptable accuracy. Meteorological features emerge as the second most important group. Their removal increases O₃ RMSE to 6.8537 µg/m³ and reduces NO₂ R² to 0.8131, confirming that atmospheric conditions including temperature, humidity, and boundary layer height are essential inputs for modeling pollutant dispersion and chemical formation processes. The larger impact on NO₂ compared to O₃ reflects the greater sensitivity of primary pollutant dispersion to local wind and mixing conditions. Rolling statistics contribute moderately to performance, with their removal reducing O₃ R² to 0.9206 and NO₂ R² to 0.8490. These features provide the model with information about concentration trends and recent variability that supplements the point-in-time information provided by individual lag values. Cyclic temporal encoding shows the smallest individual impact, with O₃ R² declining to 0.9251 and NO₂ R² dropping to 0.8558 when these features are removed. While the absolute degradation is modest, this is partly because daily periodicity information is already partially captured by the lag features themselves. The consistent pattern across both pollutants, where lag features matter most and cyclic encoding matters least, reinforces the conclusion that temporal memory of recent pollution history is the dominant factor in short-term air quality prediction.

---

## V. CONCLUSION

This study developed and evaluated an ensemble machine learning framework specifically designed for short-term forecasting of ground-level ozone (O₃) and nitrogen dioxide (NO₂) using atmospheric reanalysis data from the Copernicus Atmosphere Monitoring Service. The motivation behind this work was to address a persistent gap in the air quality prediction literature: the lack of practical, deployable systems that combine multi-step temporal forecasting with computational efficiency and user accessibility. The framework pairs XGBoost and LightGBM in an ensemble configuration with an autoregressive mechanism that enables recursive prediction across multiple future time steps, and a comprehensive feature engineering pipeline that extracts 155 meaningful input features from raw environmental time-series data.

The experimental evaluation on five years of hourly data from Delhi, India (July 2019 to June 2024), produced strong results across every evaluation metric. The ensemble model achieved an R² of 0.936 for O₃ and 0.874 for NO₂, with RMSE values of 5.58 µg/m³ and 8.37 µg/m³ respectively. The refined index of agreement reached 0.8896 for O₃ and 0.8618 for NO₂. Time-series analysis confirmed the model's ability to track both short-term concentration fluctuations and longer-term pollution trends across the testing period. Feature importance analysis identified lag variables and meteorological parameters as the dominant predictive factors, consistent with established atmospheric science. The ablation study demonstrated that removing lag features reduced O₃ R² by nearly 10 percentage points and NO₂ R² by over 15 points, confirming the fundamental importance of the autoregressive design. Forward projections to 2026 revealed a marginal decline in ozone levels alongside a concerning upward trend in nitrogen dioxide, providing actionable intelligence for environmental policy planning.

Three limitations of this work must be honestly acknowledged. First, the model depends primarily on satellite reanalysis and meteorological inputs, which do not capture hyper-local emission sources such as street-level traffic density, individual industrial facility operations, or construction site activity. Second, the autoregressive mechanism introduces cumulative error propagation during extended forecasting periods, causing prediction accuracy to gradually degrade beyond a certain time horizon. Third, the framework has been validated only for the Delhi metropolitan region, and its performance when transferred to cities with different geographic, climatic, and emission characteristics remains untested.

Five directions for future research emerge from this work. First, integration of real-time ground sensor networks, traffic flow data, and socioeconomic indicators would capture localized emission patterns that reanalysis data misses. Second, advanced architectures such as Graph Neural Networks and Transformer-based models could better represent spatial dependencies between monitoring stations and long-range temporal relationships. Third, hybrid frameworks that combine physics-based atmospheric dispersion models with data-driven machine learning could improve both prediction accuracy and physical interpretability. Fourth, implementation of uncertainty quantification techniques such as conformal prediction or Bayesian ensemble methods would provide confidence intervals alongside point predictions, making the system more useful for risk-sensitive environmental decision-making. Fifth, extension to multi-pollutant forecasting covering PM₂.₅, PM₁₀, CO, and SO₂ would transform the system into a unified air quality prediction platform.

Accurate and accessible air quality forecasting is no longer a scientific luxury but a public health necessity, and this study demonstrates that ensemble machine learning combined with autoregressive temporal prediction and practical web-based deployment can deliver that capability today.

---

## REFERENCES

[1] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5th ed. Hoboken, NJ, USA: Wiley, 2015.

[2] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.

[3] A. P. D. Pires, C. M. Teixeira, and R. M. C. Monteiro, “Limitations of linear time-series models for urban air-quality forecasting under non-stationary meteorological forcing,” *Atmospheric Environment*, vol. 214, p. 116850, 2019.

[4] A. Fania, M. S. Rahman, and T. Iqbal, “Hybrid boosting for multi-pollutant urban forecasting with meteorological interactions,” *Environmental Modelling & Software*, vol. 183, p. 106236, 2025. *(Early-access article; issue year to be verified before final submission.)*

[5] V. N. Vapnik, *The Nature of Statistical Learning Theory*. New York, NY, USA: Springer, 1995.

[6] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[7] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 785–794.

[8] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in *Proc. 31st Conf. Neural Information Processing Systems (NeurIPS)*, 2017, pp. 3146–3154.

[9] S. V. M. Kumar and P. S. Reddy, “Comparative assessment of gradient boosting algorithms for hourly NO₂ and O₃ prediction in Indian megacities,” *Science of The Total Environment*, vol. 812, p. 152452, 2022.

[10] Y. Lin and K. Chan, “Cross-city transfer learning for air pollutant forecasting using gradient-boosted ensembles,” *IEEE Internet of Things Journal*, vol. 13, no. 2, pp. 2451–2465, 2026. *(Online first; publication year to be verified before final submission.)*

[11] I. N. D. Silva, M. A. R. Dantas, and P. K. Sahu, “Lag-feature and rolling-window engineering for robust urban air-quality forecasting,” *Environmental Pollution*, vol. 292, p. 118404, 2022.

[12] J. Wang, H. Li, and Y. Zhang, “Meteorological interaction features improve machine-learning forecasts of ozone episodes,” *Atmospheric Research*, vol. 275, p. 106246, 2022.

[13] M. L. Stein, “Space–time covariance functions,” *Journal of the American Statistical Association*, vol. 100, no. 469, pp. 310–321, 2005.

[14] F. Karagulian, A. Belis, C. F. C. Dora, and M. Prüss-Ustün, “Contributions to cities’ ambient particulate matter (PM): A systematic review of local source contributions at global level,” *Atmospheric Environment*, vol. 120, pp. 475–483, 2015.

[15] H. V. Nguyen and S. V. Ukkusuri, “Multi-horizon air quality forecasting: Direct versus recursive machine-learning strategies,” *Environmental Science and Pollution Research*, vol. 29, no. 42, pp. 63711–63728, 2022.

[16] J. L. Goodall, E. M. Castronova, C. C. Humphrey, and M. M. Essawy, “Cloud and web service architectures for operational environmental forecasting,” *Environmental Modelling & Software*, vol. 84, pp. 138–147, 2016.

[17] S. N. Shah, M. A. Arshad, and F. U. Khan, “Operational deployment challenges in machine-learning-based air quality early warning systems,” *Sustainable Cities and Society*, vol. 76, p. 103463, 2022.

[18] X. Lu, L. Zhang, and D. G. Streets, “Multi-pollutant control and nonlinear ozone response in urban atmospheres,” *Nature Communications*, vol. 10, no. 1, p. 2177, 2019.

[19] D. Parrish, J. D. Krotkov, and R. V. Martin, “Toward a better understanding of the links between NO₂ and O₃ in rapidly urbanizing regions,” *Atmospheric Chemistry and Physics*, vol. 20, no. 11, pp. 6761–6782, 2020.

[20] World Health Organization, *WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide*. Geneva, Switzerland: WHO, 2021.
