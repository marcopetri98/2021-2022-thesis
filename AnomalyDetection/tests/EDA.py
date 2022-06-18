import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas._libs.tslibs import to_offset

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DATASET = "data/dataset/House1.csv"
MATPLOT_PRINT = True
USE_STL = False
LAG = 1
NLAGS = 20

def plot_acf_and_pacf(acf_values: np.ndarray,
					  pacf_values: np.ndarray,
					  acf_conf: np.ndarray,
					  pacf_conf: np.ndarray) -> None:
	lags = acf_values.shape[0] - 1
	fig, axs = plt.subplots(2, 1, figsize=(12, 12))
	axs[0].set_title("ACF function")
	axs[0].set_ylim(-1, 1)
	axs[0].plot([0, lags], [0, 0], linewidth=0.5)
	axs[0].vlines(np.linspace(0, lags, lags + 1), [0] * (lags + 1), acf_values)
	axs[0].scatter(np.linspace(0, lags, lags + 1), acf_values)
	axs[0].fill_between(np.linspace(1, lags, lags),
						acf_conf[1:, 0] - acf_values[1:],
						acf_conf[1:, 1] - acf_values[1:],
						alpha=0.25,
						linewidth=0.5)
	axs[1].set_title("PACF function")
	axs[1].set_ylim(-1, 1)
	axs[1].plot([0, lags], [0, 0], linewidth=0.5)
	axs[1].vlines(np.linspace(0, lags, lags + 1), [0] * (lags + 1), pacf_values)
	axs[1].scatter(np.linspace(0, lags, lags + 1), pacf_values)
	axs[1].fill_between(np.linspace(1, lags, lags),
						pacf_conf[1:, 0] - pacf_values[1:],
						pacf_conf[1:, 1] - pacf_values[1:],
						alpha=0.25,
						linewidth=0.5)
	plt.show()

# Exploratory data analysis
tmp_df = pd.read_csv(DATASET)
timestamps = tmp_df["Time"]
values = tmp_df["Appliance1"]

if MATPLOT_PRINT:
	plt.figure(figsize=(16, 5))
	plt.plot(timestamps, values, linewidth=0.5)
	plt.show()

test, p_value, _, _, _, _ = adfuller(values)
print("Statistical test of ADF is: ", test)
print("Computed p-value of ADF is: ", p_value)

test, p_value, _, _ = kpss(values)
print("Statistical test of KPSS is: ", test)
print("Computed p-value of KPSS is: ", p_value)

diff_tmp = np.array(values.diff(LAG))
diff_tmp = diff_tmp[LAG:]
diff_acf, diff_acf_conf = acf(diff_tmp, nlags=NLAGS, alpha=0.05)
diff_pacf, diff_pacf_conf = pacf(diff_tmp, nlags=NLAGS, alpha=0.05)

test, p_value, _, _, _, _ = adfuller(diff_tmp)
print("Statistical test of tmp_diff ADF is: ", test)
print("Computed p-value of tmp_diff ADF is: ", p_value)

test, p_value, _, _ = kpss(diff_tmp)
print("Statistical test of tmp_diff KPSS is: ", test)
print("Computed p-value of tmp_diff KPSS is: ", p_value)

if MATPLOT_PRINT:
	plt.figure(figsize=(16, 5))
	plt.plot(timestamps[LAG:], diff_tmp, linewidth=0.5)
	plt.show()
	
	plot_acf_and_pacf(diff_acf,
					  diff_pacf,
					  diff_acf_conf,
					  diff_pacf_conf)

# Try to get periodicity of the time series with pandas
period = pd.infer_freq(timestamps)
period = to_offset(period)
period = period.rule_code.upper()
print("Pandas infer a frequency of %s" % period)

if USE_STL:
	# Perform de-trending and de-seasonality
	stl = STL(values, robust=True)
	res = stl.fit()
	
	if MATPLOT_PRINT:
		fig, axs = plt.subplots(4, 1, figsize=(16, 16))
		axs[0].plot(timestamps, res.observed, linewidth=0.5)
		axs[1].plot(timestamps, res.trend, linewidth=0.5)
		axs[2].plot(timestamps, res.seasonal, linewidth=0.5)
		axs[3].scatter(timestamps, res.resid, linewidths=0.5)
		plt.show()
