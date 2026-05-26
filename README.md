# Stock Price Prediction: LSTM vs Classical ML vs ARIMA

Сравнение пяти моделей прогнозирования котировок акций (Apple, AAPL) - линейной регрессии, Random Forest, Gradient Boosting, ARIMA с walk-forward прогнозом и LSTM-нейросети. Проект расширяет популярные открытые решения на GitHub, которые используют только LSTM.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)



Вывод: на дневных данных по AAPL LSTM-нейросеть не показывает решающего преимущества над линейной регрессией и ARIMA.


За основу взяты два открытых проекта:

| Источник  | Подход |
|---|---|
| [ashendrasharma/Stock-Price-Prediction-Using-LSTM](https://github.com/ashendrasharma/Stock-Price-Prediction-Using-LSTM) |  LSTM на ценах AAPL, MinMax scaling, окно 60 дней |
| [034adarsh/Stock-Price-Prediction-Using-LSTM](https://github.com/034adarsh/Stock-Price-Prediction-Using-LSTM) |LSTM на ценах AAPL + yfinance, dropout-регуляризация |

Оба проекта используют одну модель (LSTM), цены закрытия как таргет, только MAE/RMSE как метрики и не сравнивают результат с baseline. Это оставляет открытыми вопросы:

- А действительно ли LSTM лучше простой линейной регрессии?
- А что, если просто предсказывать `P_t = P_{t-1}`?
- Почему именно нейросеть, а не классический ARIMA?

Этот проект отвечает на эти вопросы через прямое сравнение.

## Что добавлено по сравнению с источниками

| Изменение | Пояснение |
|---|---|
| Прогноз доходностей вместо цен | Tree-based модели не умеют экстраполировать выше обучающего диапазона - на ценах строят плоскую линию. На доходностях (стационарном ряде) работают корректно |
| Walk-forward для ARIMA | Многошаговый forecast(steps=N) строит почти прямую линию. Rolling-прогноз на 1 шаг устраняет это |
| 5 моделей вместо одной | Видна реальная польза каждого подхода |
| Naive baseline | Для сравнения |
| Directional Accuracy | Метрика для анализа |
| MinMax-scaling только на train | Устранена утечка из тестовой выборки|
| 18 признаков на основе доходностей | Содержательные данные вместо просто цен |


## Структура репозитория

```
.
├── stock_prediction_improved.ipynb   
├── requirements.txt                  
├── README.md                         
└── LICENSE                           
```


### Требования
- Python 3.10 или выше
- Git

### Установка

```bash
git https://github.com/GoKITTYs/ML-KAID-stock-prediction
cd ML-KAID-stock-prediction
pip install -r requirements.txt
```

### Запуск

```bash
jupyter notebook stock_prediction.ipynb
```


### Смена тикера

В ячейке `2. Загрузка данных` поменять константу `TICKER`:

```python
TICKER = 'AAPL'   # Apple - по умолчанию
# TICKER = 'GOOGL'    # Google
# TICKER = 'GAZP.ME'  # Газпром (Московская биржа)
# TICKER = 'BTC-USD'  # Bitcoin
```

## Зависимости

- `numpy`, `pandas` - обработка данных
- `matplotlib` - графики
- `yfinance` - загрузка котировок
- `scikit-learn` - Linear Regression, Random Forest, Gradient Boosting
- `statsmodels` - ARIMA, ADF-тест
- `tensorflow` - LSTM

Полный список с версиями — в `requirements.txt`.


## Источники

- [ashendrasharma/Stock-Price-Prediction-Using-LSTM](https://github.com/ashendrasharma/Stock-Price-Prediction-Using-LSTM) - базовая LSTM-архитектура
- [034adarsh/Stock-Price-Prediction-Using-LSTM](https://github.com/034adarsh/Stock-Price-Prediction-Using-LSTM) - подход через `yfinance` и dropout


MIT License. См. [LICENSE](LICENSE).
