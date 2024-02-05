#!/usr/bin/env python
# coding: utf-8

# # Statistics2

# In[17]:


# Импортируем библиотеки

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')


# In[19]:


# Удаляем ненужный столбец
data.drop('Index', axis = 1, inplace = True)
data.head()


# In[20]:


data.shape


# ## 1. Создайте колонку BMI (индекс массы тела), которая считается по формуле: BMI = (Вес в кг) / (Рост в метрах)^2. Учтите, что ваш текущий рост в см, не метры.

# In[21]:


data['Height_m'] = data['Height'] / 100

data['BMI'] = data['Weight'] / (data['Height_m'] ** 2)

print(data)


# ## 2. Посчитайте среднее значение, медиану, и моду для роста, веса, BMI.

# In[22]:


print(f"Среднее значение роста: {data['Height'].mean()}")
print(f"Медиана роста: {data['Height'].median()}")
print(f"Мода роста: {data['Height'].mode().values[0]}")

print(f"\nСреднее значение веса: {data['Weight'].mean()}")
print(f"Медиана веса: {data['Weight'].median()}")
print(f"Мода веса: {data['Weight'].mode().values[0]}")

print(f"\nСреднее значение BMI: {data['BMI'].mean()}")
print(f"Медиана BMI: {data['BMI'].median()}")
print(f"Мода BMI: {data['BMI'].mode().values[0]}")


# In[ ]:





# ## 3. Найдите range (диапазон, т.е. макс - мин), и стандартное отклонение для роста, веса, BMI. 

# In[23]:


print(f"Диапазон роста: {np.ptp(data['Height'])} sm  ")
print(f"Стандартное отклонение роста: {data['Height'].std()} sm ")

print(f"\nДиапазон веса: {np.ptp(data['Weight'])} kg")
print(f"Стандартное отклонение веса: {data['Weight'].std()} kg")

print(f"\nДиапазон BMI: {np.ptp(data['BMI'])}")
print(f"Стандартное отклонение BMI: {data['BMI'].std()}")


# In[ ]:





# ## 4. Найдите 15-й процентиль и 90-й процентиль для роста, веса, BMI. 

# In[24]:


print(f"15-й процентиль роста: {np.percentile(data['Height'], 15)} см")
print(f"90-й процентиль роста: {np.percentile(data['Height'], 90)} см")


# In[25]:


print(f"\n15-й процентиль веса: {np.percentile(data['Weight'], 15)} кг")
print(f"90-й процентиль веса: {np.percentile(data['Weight'], 90)} кг")


# In[26]:


print(f"\n15-й процентиль BMI: {np.percentile(data['BMI'], 15)}")
print(f"90-й процентиль BMI: {np.percentile(data['BMI'], 90)}")


# ## 5. Создайте гистограмму для роста, веса, BMI.

# In[31]:


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(data['Height'], bins=20, color='blue', alpha=0.7, rwidth=0.8)
plt.title('Распределение Роста')
plt.xlabel('Рост (см)')
plt.ylabel('Частота')

plt.subplot(1, 3, 2)
plt.hist(data['Weight'], bins=20, color='green', alpha=0.7, rwidth=0.8)
plt.title('Распределение Веса')
plt.xlabel('Вес (кг)')
plt.ylabel('Частота')

plt.subplot(1, 3, 3)
plt.hist(data['BMI'], bins=20, color='red', alpha=0.7, rwidth=0.8)
plt.title('Распределение BMI')
plt.xlabel('BMI')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()


# In[ ]:





# ## 6. Создайте box plot для роста, веса, BMI в разрезе пола. 

# In[36]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='Gender', y='Height', data=data)
plt.title('Box Plot Роста')

plt.tight_layout()
plt.show()


# In[34]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 2)
sns.boxplot(x='Gender', y='Weight', data=data)
plt.title('Box Plot Веса')

plt.tight_layout()
plt.show()


# In[35]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 3)
sns.boxplot(x='Gender', y='BMI', data=data)
plt.title('Box Plot BMI')

plt.tight_layout()
plt.show()


# In[ ]:


#данные по росту и весу не имеют выбросов, значения в наборе данных распределены
#более равномерно и нет необычных или значительно отклоняющихся от среднего значений


# ## 7. Посчитайте межквартильный размах для роста, веса, BMI.

# In[37]:


iqr_height = np.percentile(data['Height'], 75) - np.percentile(data['Height'], 25)
iqr_weight = np.percentile(data['Weight'], 75) - np.percentile(data['Weight'], 25)
iqr_bmi = np.percentile(data['BMI'], 75) - np.percentile(data['BMI'], 25)

print(f"iqr для роста: {iqr_height} см")
print(f"iqr для веса: {iqr_weight} кг")
print(f"iqr для BMI: {iqr_bmi}")


# In[ ]:





# In[ ]:





# ## 8. Есть ли в данных какие-то выбросы для роста, веса, BMI? Как можно их найти? Используя межквартильный размах? Удалите все такие выбросы из таблицы.

# In[39]:


def remove_outliers(column):
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return [value if lower_limit <= value <= upper_limit else np.nan for value in column]

data['BMI'] = remove_outliers(data['BMI'])

df = pd.DataFrame(data)

df = df.dropna()

print("Данные без выбросов для столбца 'BMI':")
print(df)


# In[40]:


print(data)


# ______________________________
