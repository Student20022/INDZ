import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from data import df


def pair_regression():
    X = df['Торгівельна площа']
    y = df['Товарообіг']
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(model.summary())

    plt.scatter(df['Торгівельна площа'], df['Товарообіг'], color='blue')
    plt.plot(df['Торгівельна площа'], model.predict(X), color='red')
    plt.xlabel('Торгівельна площа (тис. м²)')
    plt.ylabel('Річний товарообіг (млн. грн.)')
    plt.title('Парна регресія: Товарообіг vs Торгівельна площа')
    plt.show()

    correlation = df['Торгівельна площа'].corr(df['Товарообіг'])
    r2 = r2_score(df['Товарообіг'], model.predict(X))

    print(f'Коефіцієнт кореляції: {correlation}')
    print(f'Коефіцієнт детермінації (R²): {r2}')

    X_new = np.array([1, 1200])
    forecast = model.predict(X_new)
    print(f'Прогноз річного товарообігу: {forecast[0]:.2f} млн. грн.')

    alpha = 0.05  # надійність 95%
    predictions = model.get_prediction(X_new)
    conf_int = predictions.conf_int(alpha=alpha)
    print(f'Інтервал прогнозу: {conf_int}')

    forecast_with_factors = (
        forecast[0] + (15000 / 1000) * 0.03 + (10000 / 1000) * 0.05
    )
    print(
        f'Прогноз річного товарообігу з врахуванням 15000 осіб/день та 10000 грн/день: {forecast_with_factors:.2f} млн. грн.'
    )
