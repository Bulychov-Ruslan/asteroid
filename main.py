# %%
from analysis.data_analysis import explore_data
from analysis.hypothesis import hypothesis_test
from analysis.regression import perform_regression_analysis
from utils.helper_functions import load_dataset


# %%
# Загрузка данных
dataset = load_dataset("data/asteroid.csv")

# %%
# Проведение предварительного анализа данных
EDA = explore_data(dataset)

# %%
# Проверка гипотез
hypothesis_test(EDA)

# %%
# Регрессионный анализ
perform_regression_analysis(EDA)



# %%
