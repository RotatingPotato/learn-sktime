# sktime
> [name=Chien-Hsun, Chang & Kuo-Wei, Wu] 
> National Taichung University of Science and Technology, Taichung, Taiwan.

**[ 目錄 ]**
> [TOC]

### 什麼是 sktime
sktime 是 Python 中的時間序列分析套件。它提供了一個統一的介面，適用於多種時間序列分析。目前，這包括時間序列分類、回歸、聚類、標註和預測。它附帶了時間序列演算法和與 scikit-learn 兼容的工具，用於建立、改良和驗證時間序列模型(Löning et al., 2022)。
![sktime](https://hackmd.io/_uploads/SyxWCCkba.png)


### sktime 和 Scikit-learn 的關係
sktime 是基於 scikit-learn API 的時間序列機器學習工具套件。它擴展了 scikit-learn 的功能，使其能夠處理時間序列數據。sktime 提供了專用的時間序列學習演算法和轉換方法，而其他常見套件中尚不支援。此外，sktime 旨在與 scikit-learn 互相操作，可輕鬆地將演算法修改為相關的時間序列，並構建複合模型。

#### 為什麼不能單純用 Scikit-learn ？
Scikit-learn 是一個通用的機器學習套件，它並不完全適合處理時間序列數據。Scikit-learn 假設數據以表格格式進行結構化，並且每一列均為獨立同分布（i.i.d, 時間序列數據不適用的假設。）。同時，Scikit-learn 缺少許多重要的時間序列操作，例如：跨時間將數據拆分為訓練集和測試集。因此使用 sktime 可以解決 Scikit-learn 無法容易處理時間序列數據的問題。

### 機器學習
機器學習(Machine Learning)，簡單來說，就是讓機器去學習，機器要如何去學習呢?
1. 篩選出正確需要的資料
2. 將資料分類
3. 訓練資料(包含訓練及測試兩階段)
4. 將訓練完成的知料模型來預測未來資料

## 使用 sktime 做時間序列迴歸
隨著 [sktime 文件](https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples)測試並練習使用 sktime 套件
+ [GitHub Repo](https://github.com/RotatingPotato/learn-sktime)

### 時間序列分析
![](https://hackmd.io/_uploads/SJRNskx-6.png)

#### 1. 安裝 sktime
```bash
$ pip install sktime
```

#### 2. 匯入套件
```python
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 隱藏錯誤訊息
warnings.filterwarnings("ignore")
```

#### 3. 選擇資料
這裡我們使用 sktime 內建的練習用資料集 `airline` 練習。
```python
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
y = load_airline()
```
現在把資料集圖表輸出出來看看。
```python
# 顯示資料
plot_series(y)
plt.show()
print(y.index)
```
輸出：
![](https://hackmd.io/_uploads/By4iA1eWa.png)
```output
PeriodIndex(['1949-01', '1949-02', '1949-03', '1949-04', '1949-05', '1949-06',
             '1949-07', '1949-08', '1949-09', '1949-10',
             ...
             '1960-03', '1960-04', '1960-05', '1960-06', '1960-07', '1960-08',   
             '1960-09', '1960-10', '1960-11', '1960-12'],
            dtype='period[M]', length=144)
```
我們將時間序列表示為 pd.series，其中索引代表時間點。sktime 支援 pandas 的整數、週期和時間戳記。在這個練習中，我們使用的是週期索引（PeriodIndex）。

#### 4. 切割資料
將資料集切割成 `訓練集` 與 `測試集`。
```python
# 將資料集切割成訓練集與測試集
from sktime.forecasting.model_selection import temporal_train_test_split
y_train, y_test = temporal_train_test_split(y, test_size=36)
```
+ 我們將嘗試預測最近 3 年的資料，使用前幾年的資料作為訓練資料。系列中的每個點代表一個月，因此我們應該保留最後 36 個點作為測試資料，並使用 36 步的超前預測範圍來評估預測效能。
+ 我們將使用 sMAPE（對稱平均絕對誤差百分比）來量化我們預測的準確度。較低的 sMAPE 意味著較高的準確度。

同樣將資料集圖表輸出出來看看。
```python
# 顯示資料
plot_series(y_train, y_test, labels=["y_train", "y_test"])
plt.show()
```
輸出：
![](https://hackmd.io/_uploads/HJA38lg-p.png)
可以發現訓練集與測試集已經切割好了。


#### 5. 預測範圍
當我們要進行預測時，我們要指定預測的範圍，並傳遞給我們所指定的演算法。
+ **5-1. 相對預測範圍**

    因為我們感興趣的是第 1 步到第 36 步的預測（剛剛所分割出來的範圍），所以程式碼如下：
    ```python
    # 相對預測範圍
    fh = np.arange(len(y_test)) + 1
    print(fh)
    ```
    輸出：
    ```output
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36]
    ```
    當然，我們也可以指定不同的範圍例如只預測前面的第 2 步和第 5 步：
    ```python
    fh = np.array([2, 5])
    print(fh)
    ```
    輸出：
    ```output
    [2 5]
    ```
+ **5-2. 絕對預測範圍**
    
    我們也可以使用絕對時間來指定預測範圍，要這麼做的話要使用 `sktime` 的 `ForecastingHorizon`。這樣就能簡單的從測試集中的時間點創立預測範圍。 
    ```python
    from sktime.forecasting.base import ForecastingHorizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    print(fh)
    ```
    輸出：
    ```output
    ForecastingHorizon(['1958-01', '1958-02', '1958-03', '1958-04', '1958-05', '1958-06',
             '1958-07', '1958-08', '1958-09', '1958-10', '1958-11', '1958-12',   
             '1959-01', '1959-02', '1959-03', '1959-04', '1959-05', '1959-06',   
             '1959-07', '1959-08', '1959-09', '1959-10', '1959-11', '1959-12',   
             '1960-01', '1960-02', '1960-03', '1960-04', '1960-05', '1960-06',   
             '1960-07', '1960-08', '1960-09', '1960-10', '1960-11', '1960-12'],  
            dtype='period[M]', is_relative=False)
    ```

#### 6. 進行預測
和 Scikit-learn 相同，預測前我們要先指定（或建立）模型，然後將其擬合到訓練資料中，最後呼叫 `predict` 來產生指定範圍的預測。
不同的是 `sktime` 內建了幾種預測的演算法（forecasters）和建立綜合模型的工具。所有 forecasters 都有一個共同介面。forecasters 根據單一系列資料進行訓練，再對提供的預測範圍做預測，我們採用採用兩種預測策略作為練習。
#### 6-1. 預測最後的數值
```python
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
# 預測最後的數值
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
```
把結果顯示出來看看：
```python
# 顯示資料
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plt.show()
```
輸出：
![](https://hackmd.io/_uploads/HkAza6Ibp.png)

接著我們來做 sMAPE 的計算，sMAPE 的用途是評估預測模型的精準度。
```python
# 計算sMAPE
sMAPE = mean_absolute_percentage_error(y_pred, y_test)
print(sMAPE)
```
輸出：
```output
0.2825727513227514
```

#### 6-2. 預測同季最後的數值
```python
# 預測同季最後的數值
forecaster = NaiveForecaster(strategy="last", sp=12)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
```
同樣把結果顯示出來看看並計算 sMAPE：
```python
# 顯示資料
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plt.show()

# 計算 sMAPE
sMAPE = mean_absolute_percentage_error(y_pred, y_test)
print(sMAPE)
```
輸出：
![](https://hackmd.io/_uploads/S1mLxCLWp.png)
```output
0.1625132136966463
```

### 完整程式碼如下：
```python=
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
```

## 提高預測精準度
當然，要完成有效的預測我們就需要更加的精準，接下來我們嘗試使用 sktime 中不同的統計演算法來做繼續預測。
![](https://hackmd.io/_uploads/Hy_uERLZa.png)

### 

## 參考文獻
> 1. Markus Löning, Franz Király, Tony Bagnall, Matthew Middlehurst, Sajaysurya Ganesh, George Oastler, Jason Lines, Martin Walter, ViktorKaz, Lukasz Mentel, chrisholder, Leonidas Tsaprounis, RNKuhns, Mirae Parker, Taiwo Owoseni, Patrick Rockenschaub, danbartl, jesellier, eenticott-shell, … Beth rice. (2022). sktime/sktime: v0.13.4 (v0.13.4). Zenodo. https://doi.org/10.5281/zenodo.7117735

