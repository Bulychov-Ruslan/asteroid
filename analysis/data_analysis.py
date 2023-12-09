import pandas as pd
import seaborn as sns                   
import matplotlib.pyplot as plt   


def explore_data(df):
    # Ваш код для предварительного анализа данных здесь

    # Вывод первых строк датасета
    print("Первые строки датасета:")
    print(df.head())

    # Вывод последних строк датасета
    print("\nПоследние строки датасета:")
    print(df.tail())

    # Информация о датасете
    print("\nИнформация о датасете:")
    print(df.info())

    # Необходимые данные
    columns_to_drop = ['id', 'spkid', 'name', 'prefix', 'orbit_id', 'epoch_mjd', 'equinox',
                       'sigma_e', 'sigma_a', 'sigma_q', 'sigma_i', 'sigma_om', 'sigma_w',
                       'sigma_ma', 'sigma_ad', 'sigma_n', 'sigma_tp', 'sigma_per', 'rms']
    df = df.drop(columns=columns_to_drop)

    # Форма датасета
    print("\nФорма датасета после удаления ненужных столбцов:")
    print(df.shape)

    # Удаление дубликатов
    df = df.drop_duplicates()
    print("\nФорма датасета после удаления дубликатов:")
    print(df.shape)

    # Количество пропущенных значений
    print("\nКоличество пропущенных значений по столбцам:")
    print(df.isnull().sum())

    # Замена NULL-значений
    df['neo'].fillna(df['neo'].mode().iloc[0], inplace=True)
    df['pha'].fillna(df['pha'].mode().iloc[0], inplace=True)
    columns_to_replace_zero = ['diameter', 'albedo', 'diameter_sigma']
    df[columns_to_replace_zero] = df[columns_to_replace_zero].fillna(0)
    df['H'].fillna(df['H'].mean(), inplace=True)
    df['ma'].fillna(df['ma'].mean(), inplace=True)
    df['ad'].fillna(df['ad'].mean(), inplace=True)
    df['per'].fillna(df['per'].mean(), inplace=True)
    df['per_y'].fillna(df['per_y'].mean(), inplace=True)
    df['moid'].fillna(df['moid'].mean(), inplace=True)
    df['moid_ld'].fillna(df['moid_ld'].mean(), inplace=True)

    # Количество пропущенных значений после обработки
    print("\nКоличество пропущенных значений после обработки:")
    print(df.isnull().sum())

    # Описательная статистика
    print("\nОписательная статистика:")
    print(df.describe().T)

    # Распределение числовых переменных
    print("\nРаспределение числовых переменных:")
    df.hist(bins=20, figsize=(15, 10))
    plt.show()

    # Исследование категориальных переменных
    print("\nИсследование категориальных переменных:")
    categorical_columns = df.select_dtypes(include=['category', 'object']).columns
    for column in categorical_columns:
        print(f"Уникальные значения для {column}:\n{df[column].value_counts()}\n")

    # Круговые диаграммы для категориальной переменной 'class'
    print("\nКруговая диаграмма для переменной 'class':")
    plt.figure(figsize=(8, 8))
    df['class'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=90)
    plt.title("Распределение по классам")
    plt.ylabel("")
    plt.show()

    p = df.hist(figsize = (20,20)) # гистограмма

    plt.figure(figsize=(25,25))
    p = sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='RdYlGn') # матрица корреляции

    return df