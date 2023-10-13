import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# 隱藏錯誤訊息
warnings.filterwarnings("ignore")

# 選擇資料
y = load_airline()

# 顯示資料
plot_series(y)
plt.show()
print(y.index)

# 將資料集切割成訓練集與測試集
y_train, y_test = temporal_train_test_split(y, test_size=36)

# 顯示資料
plot_series(y_train, y_test, labels=["y_train", "y_test"])
plt.show()

# 相對預測範圍
fh = np.arange(len(y_test)) + 1
print(fh)

# 絕對預測範圍
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(fh)

# 預測最後的數值
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

# 顯示資料
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plt.show()

# 計算 sMAPE
sMAPE = mean_absolute_percentage_error(y_pred, y_test)
print(sMAPE)

# 預測同季最後的數值
forecaster = NaiveForecaster(strategy="last", sp=12)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

# 顯示資料
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plt.show()

# 計算 sMAPE
sMAPE = mean_absolute_percentage_error(y_pred, y_test)
print(sMAPE)