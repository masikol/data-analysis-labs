import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
import lightgbm as lgb

# Загружаем входные данные
infpath = 'Cleaned-Data_covid.csv'
df = pd.read_csv(
    infpath
)

# Классы несбалансированы:
#  Severity_Severe=0 -- 237600 случаев
#  Severity_Severe=1 -- 79200 случаев
# Ерунда, но логистическая регрессия в таком случае у нас просто проставляло всем
#   проверочным данным нули (Severity_Severe=0)
# Поэтому, чтобы упростить задачу логистической регрессии, мы в лоб сбаллансируем датафрейм:
#   возьмём и тех, и тех по 79200 штук.

# Для удобства
df['case_id'] = df.index

# Предикторы. Независимые переменные.
# На основании общей эрудиции именно эти переменные позволят отличить тяжёлый случай от не-тяжёлого
predictor_cols = [
    'Fever',
    'Dry-Cough',
    'Age_60+',
    'Difficulty-in-Breathing',
    'None_Sympton',
    'None_Experiencing',
    'Gender_Female',
]

# Зависимая переменная -- её предсказываем
target_col = 'Severity_Severe'

classif_df = df[predictor_cols + [target_col]]

min_sample_size = min(
    df.groupby(target_col, as_index=False) \
        .agg({'case_id': 'count'})['case_id']
)

classif_df = classif_df[classif_df[target_col] == 1].sample(min_sample_size) \
    .append(
        classif_df[classif_df[target_col] == 0].sample(min_sample_size)
    )

# Вот теперь в classif_df у нас по 79200 "тяжёлых" и "не тяжёлых"


# Сформируем обучающую и проверочную выборки.
# Почему тестировочных -- четверть... обучающих точно надо побольше, чем тестовых.
# А на сколько -- ну, не понятно. Почему бы и не четверть.
x_train, x_test, y_train, y_test = train_test_split(
    classif_df[predictor_cols],
    classif_df[target_col],
    test_size=0.25,
    random_state=0
)

# |=== Логистическая регрессия ===|

# Параметры по умолчаюнию
logisticRegr = LogisticRegression()

# Замеряем время обучения
%%timeit
logisticRegr.fit(np.array(x_train), np.array(y_train).ravel())
# Output:
#   219 ms ± 14.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Обучаем модель для того, чтобы проверить точность.
# Почему-то изнутри timeit'а обученную модель не получилось достать
logisticRegr.fit(np.array(x_train), np.array(y_train).ravel())

# Замеряем время классификации
%%timeit
logit_prediction = logisticRegr.predict(
    np.array(x_test)
)
# Output:
#   2.2 ms ± 115 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Классифицируем
# Почему-то изнутри timeit'а предсказанные значения не получилось достать
logit_prediction = logisticRegr.predict(
    np.array(x_test)
)

# Рассчитываем специфичность, чувствительность, F-меру
print(precision_score(y_test, logit_prediction))
# Output:
#   0.4929441201000834
print(recall_score(y_test, logit_prediction))
# Output:
#   0.7556123753515724
print(f1_score(y_test, logit_prediction))
# Output:
#   0.5966484958610943

# Помня про несбаллансированные выборки, на всякий случай проверяем,
#   не проставила ли модель и тут всем тестовым данным просто нули или единицы
set(logit_prediction)
# Output:
#   {0, 1}
# Вот теперь тут не только нули, как было с несбаллансированными данными


# |=== XGboost ===|

# Создаём модельку
# Много всяких параметров...
# Взяли просто как в туториале
xg_reg = xgb.XGBClassifier(
    objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
    max_depth = 5, alpha = 10, n_estimators = 100, use_label_encoder=False
)

# Замеряем время обучения
%%timeit
xg_reg.fit(
    np.array(x_train),
    np.array(y_train)
)
# Output:
#   812 ms ± 157 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Обучаем модель для того, чтобы проверить точность.
# Почему-то изнутри timeit'а обученную модель не получилось достать
xg_reg.fit(
    np.array(x_train),
    np.array(y_train)
)

# Замеряем время классификации
%%timeit
xg_prediction = xg_reg.predict(
    np.array(x_test)
)
# Output:
#   15.6 ms ± 755 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Классифицируем
xg_prediction = xg_reg.predict(
    np.array(x_test)
)

# Рассчитываем специфичность, чувствительность, F-меру
print(precision_score(y_test, xg_prediction))
# Output:
#   0.490790862356586
print(recall_score(y_test, xg_prediction))
# Output:
#   0.7372027614420864
print(f1_score(y_test, xg_prediction))
# Output:
#   0.5892740353172008

# Помня про несбаллансированные выборки, на всякий случай проверяем,
#   не проставила ли модель и тут всем тестовым данным просто нули или единицы
set(xg_prediction)
# Output:
#   {0, 1}


# |=== LightGBM ===|

# Создаём модельку
# Взяли параметры просто как в туториале
lgb_classifier = lgb.LGBMClassifier(
    learning_rate=0.09,
    max_depth=-5,
    random_state=42
)

# Замеряем время обучения
%%timeit
lgb_classifier.fit(
    x_train,
    y_train,
    eval_set=[(x_test,y_test),(x_train,y_train)],
    eval_metric='logloss'
)
# Output:
#   918 ms ± 111 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Обучаем модель для того, чтобы проверить точность.
# Почему-то изнутри timeit'а обученную модель не получилось достать
lgb_classifier.fit(
    x_train,
    y_train,
    eval_set=[(x_test,y_test),(x_train,y_train)],
    eval_metric='logloss'
)

# Замеряем время классификации
%%timeit
lbg_prediction = lgb_classifier.predict(x_test)
# Output:
#   70.3 ms ± 3.96 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Классифицируем
lbg_prediction = lgb_classifier.predict(x_test)\

# Рассчитываем специфичность, чувствительность, F-меру
print(precision_score(y_test, lbg_prediction))
# Output:
#   0.48803038990825687
print(recall_score(y_test, lbg_prediction))
# Output:
#   0.6963947839427257
print(f1_score(y_test, lbg_prediction))
# Output:
#   0.5738848269032217

# Помня про несбаллансированные выборки, на всякий случай проверяем,
#   не проставила ли модель и тут всем тестовым данным просто нули или единицы
set(prediction)
# Output:
#   {0, 1}

