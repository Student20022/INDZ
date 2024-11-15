import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data import df


def multi_factor_regression():
    X_multi = df[
        ['Торгівельна площа', 'Інтенсивність потоку', 'Середньоденний дохід']
    ]
    y = df['Товарообіг']

    X_multi = sm.add_constant(X_multi)

    vif_data = pd.DataFrame()
    vif_data['feature'] = X_multi.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X_multi.values, i)
        for i in range(X_multi.shape[1])
    ]

    print('VIF для кожного фактора:')
    print(vif_data)

    X_multi = X_multi.drop(columns=['Інтенсивність потоку'])

    model_multi = sm.OLS(y, X_multi).fit()
    print(model_multi.summary())

    r2_multi = r2_score(y, model_multi.predict(X_multi))
    print(f'Коефіцієнт детермінації (R²): {r2_multi}')

    elasticity = model_multi.params[1:] * (
        df[['Торгівельна площа', 'Середньоденний дохід']].mean()
        / df['Товарообіг'].mean()
    )
    print('Частинні коефіцієнти еластичності:')
    print(elasticity)

    X_new_multi = np.array(
        [1, 1200, 10000]
    )
    forecast_multi = model_multi.predict(X_new_multi)
    print(f'Прогноз річного товарообігу: {forecast_multi[0]:.2f} млн. грн.')

    predictions_multi = model_multi.get_prediction(X_new_multi)
    conf_int_multi = predictions_multi.conf_int(alpha=0.05)
    print(f'Інтервал прогнозу: {conf_int_multi}')
