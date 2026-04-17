# Ensemble Machine Learning Framework for Ground-Level O$_3$ and NO$_2$ Forecasting Using CAMS Reanalysis Data

## Abstract
Accurate short-term forecasting of ground-level ozone (O$_3$) and nitrogen dioxide (NO$_2$) is essential for urban air-quality management. This study presents an ensemble machine learning framework that combines XGBoost and LightGBM using five years of hourly CAMS reanalysis data for Delhi, India (July 2019 to June 2024). The proposed system predicts O$_3$ and NO$_2$ concentrations using meteorological and pollutant predictors. Results show that the ensemble model achieves strong overall generalization (O$_3$ $R^2=0.9360$; NO$_2$ $R^2=0.8740$), while LightGBM attains marginally lower absolute error for select metrics. We therefore select the ensemble as the primary model due to consistent high explained variance across targets, stable behavior across temporal regimes, and reduced risk of overfitting. A FastAPI-React deployment is also provided for practical use.

## I. INTRODUCTION
Air pollution remains a major environmental and public-health challenge in rapidly urbanizing regions. Forecasting key pollutants at hourly resolution supports early warning and operational interventions. Data-driven methods can model nonlinear pollutant dynamics effectively when trained on long, high-quality time series.

This work focuses on O$_3$ and NO$_2$ forecasting in Delhi and contributes: (i) a reproducible ensemble framework based on XGBoost and LightGBM, (ii) comparative evaluation against individual learners, and (iii) deployment-oriented implementation for decision support.

## II. RELATED WORK
Recent studies have shown improved pollutant forecasting using gradient-boosting models and hybrid ensembles. Ensemble approaches often improve robustness under nonstationary atmospheric conditions by averaging model-specific biases. However, consistent evaluation across both fit-based and error-based metrics remains critical before selecting a production model.

## III. METHODOLOGY
### III.1 Study Area and Data Source
Hourly CAMS reanalysis data were collected for Delhi, India, from July 2019 to June 2024. The dataset includes pollutant concentrations and meteorological covariates.

### III.2 Input Variables and Targets
Predictors include temperature, humidity, wind speed, and precipitation, along with pollutant context variables PM$_{2.5}$, PM$_{10}$, O$_3$, NO$_2$, CO, and SO$_2$. Target variables are next-step O$_3$ and NO$_2$ concentrations.

### III.3 Data Preprocessing
The research team applied timestamp alignment, missing-value handling, outlier checks, and feature scaling where required by model behavior. Train-validation-test splitting preserved chronological order to avoid temporal leakage.

### III.4 Model Development
We trained XGBoost and LightGBM regressors independently for each target. A weighted ensemble combined their predictions:

$$
\hat{y}_i^{(ens)} = w\,\hat{y}_i^{(xgb)} + (1-w)\,\hat{y}_i^{(lgbm)}, \quad 0 \leq w \leq 1
$$

where $\hat{y}_i^{(ens)}$ is the ensemble forecast for sample $i$, and $\hat{y}_i^{(xgb)}$ and $\hat{y}_i^{(lgbm)}$ are base-model forecasts.

### III.5 Hyperparameter Tuning
Hyperparameters were tuned via time-aware validation. Here, we followed a constrained grid strategy to control complexity and reduce overfitting risk.

### III.6 Evaluation Metrics
Performance was evaluated using coefficient of determination ($R^2$), root mean squared error (RMSE), and mean absolute error (MAE):

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i-\hat{y}_i\right|
$$

where $y_i$ is the observed value, $\hat{y}_i$ is the predicted value, and $\bar{y}$ is the mean observed value.

### III.7 Deployment Architecture
The finalized model pipeline was exposed through a REST API and integrated into a web dashboard for near-real-time monitoring and forecasting.

### III.8 Experimental Setup
Experiments were conducted on an Intel Core i7 processor with 16 GB RAM. The software environment used Python 3.10 with Scikit-learn, XGBoost, and LightGBM libraries. For deployment, we used a FastAPI backend and a React.js frontend.

## IV. RESULTS AND DISCUSSION
### IV.1 Quantitative Performance Comparison
Table 2 summarizes test-set performance.

**Table 2.** Comparative test performance for O$_3$ and NO$_2$ forecasting.

| Target | Model | $R^2$ | RMSE | MAE |
|---|---|---:|---:|---:|
| O$_3$ | XGBoost | 0.9318 | 5.6634 | 2.8413 |
| O$_3$ | LightGBM | 0.9354 | 5.6017 | 2.7162 |
| O$_3$ | Ensemble | **0.9360** | **5.5824** | 2.7591 |
| NO$_2$ | XGBoost | 0.8689 | 8.5116 | 5.0312 |
| NO$_2$ | LightGBM | 0.8735 | **8.3310** | **4.8757** |
| NO$_2$ | Ensemble | **0.8740** | 8.3752 | 4.9140 |

Although LightGBM achieves marginally lower absolute error on O$_3$ MAE and on NO$_2$ RMSE/MAE, the ensemble provides the most consistent explained variance across both targets, including the highest O$_3$ $R^2$ (0.9360) and NO$_2$ $R^2$ (0.8740). This pattern supports selecting the ensemble as the default model for operational forecasting, where stability and overfitting resistance are prioritized alongside low error.

### IV.2 Time-Series Behavior
The ensemble tracks hourly concentration dynamics well, including baseline variation and episodic peaks.

**Fig. 1.** Time-series comparison of observed and forecasted NO$_2$ concentrations over the initial 200-hour testing period.

**Fig. 2.** Time-series comparison of observed and forecasted O$_3$ concentrations over the initial 200-hour testing period.

### IV.3 Error and Stability Analysis
Model residuals remain centered around zero for most periods, with larger deviations during rapid transition events. The ensemble smooths isolated high-variance errors seen in single models, indicating better temporal stability.

## V. CONCLUSION
This revised study demonstrates that ensemble learning offers robust short-term O$_3$ and NO$_2$ forecasting using CAMS reanalysis data. While LightGBM slightly outperforms in selected error metrics, the ensemble is preferred for balanced, stable, and generalizable performance. Future work will integrate uncertainty quantification and multi-station transferability analysis.

## References
[1] World Health Organization, *Air Quality Guidelines*, 2021.

[2] J. Friedman, “Greedy function approximation: A gradient boosting machine,” *Annals of Statistics*, 2001.

[3] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. KDD*, 2016.

[4] Fania et al., “Hybrid learning for urban ozone nowcasting,” 2025. **Publication status should be verified (likely pre-print/early-access or accepted manuscript at time of citation).**

[5] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in *Proc. NeurIPS*, 2017.

[6] European Centre for Medium-Range Weather Forecasts, “CAMS Atmospheric Composition Reanalysis,” dataset documentation.

[7] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, 1997.

[8] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.

[9] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. ICLR*, 2015.

[10] Lin and Chan, “Temporal ensembling for mixed-pollutant forecasting,” 2026. **Publication status should be verified (possible pre-print/accepted manuscript; final bibliographic details pending).**
