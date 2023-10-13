import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 隱藏錯誤訊息
warnings.filterwarnings("ignore")

# 選擇資料
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
y = load_airline()

# 顯示資料
plot_series(y)
plt.show()
print(y.index)

# 將資料集切割成訓練集與測試集
from sktime.forecasting.model_selection import temporal_train_test_split
y_train, y_test = temporal_train_test_split(y, test_size=36)

# 顯示資料
plot_series(y_train, y_test, labels=["y_train", "y_test"])
plt.show()

# 相對預測範圍
fh = np.arange(len(y_test)) + 1
print(fh)

# 絕對預測範圍
from sktime.forecasting.base import ForecastingHorizon
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(fh)

# 預測最後的數值
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import smape_loss
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plt.show()
smape_loss(y_pred, y_test)