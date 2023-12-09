import pandas as pd
import seaborn as sns                   
import matplotlib.pyplot as plt   
from scipy.stats import ttest_ind


def hypothesis_test(df):

    omb_data = df[df['class'].str.upper() == 'OMB']['diameter'].dropna()
    mba_data = df[df['class'].str.upper() == 'MBA']['diameter'].dropna()

    std_omb = omb_data.std()
    std_mba = mba_data.std()

    print(f'Стандартное отклонение OMB: {std_omb}')
    print(f'Стандартное отклонение MBA: {std_mba}')

    t_stat, p_value = ttest_ind(omb_data, mba_data, equal_var=False)

    print(f'T-статистика: {t_stat}')
    print(f'P-значение: {p_value}')

    alpha = 0.05
    if p_value < alpha:
        print('Отвергаем нулевую гипотезу. Существует статистически значимая разница в средних значениях диаметров.')
    else:
        print('Не удалось отвергнуть нулевую гипотезу. Нет статистически значимой разницы в средних значениях диаметров.')

    # Гипотеза: Средние диаметры астероидов различаются в зависимости от их классификации.
