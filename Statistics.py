#!/usr/bin/env python
# coding: utf-8

# # Описательная статистика

# ## Часть 1 - чистые данные

# Библиотека seaborn работает с некоторыми известными датасетами, на которых мы можем опробовать описательный анализ данных.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# датасет mpg, описывающий некоторые из машин бывших в продаже на американском рынке. Довольно интересная информация здесь - характеристика mpg - miles per gallon, сколько миль автомобиль может проехать на галлоне бензина.

# In[2]:


data=sns.load_dataset('mpg')


# In[10]:


data.head(10)


# ## Задача 1

# ### 1.1 Посчитайте среднее значение mpg 

# In[6]:


data['mpg'].mean()


# ### 1.2 Посчитайте медиану mpg

# In[7]:


data['mpg'].median()


# ## Задача 2

# ### Нарисуйте гистограмму величины мpg, используя seaborn.

# In[11]:


sns.histplot(data['mpg'], kde=False, bins=20)  
plt.title('Гистограмма для mpg')
plt.xlabel('mpg')
plt.ylabel('Частота')
plt.show()


# ## Задача 3

# ### 3.1 Нарисуйте 3 ящика с усами, показывающие распределение mpg в зависимости от региона, в котором машина была произведена

# Используйте boxplot, давая две переменные х и y. Если вам нужна будет помощь, посмотрите информацию по ссылке: https://seaborn.pydata.org/examples/grouped_boxplot.html

# In[16]:


sns.boxplot(x='origin', y='mpg', data=data)
plt.title('Распределение mpg в зависимости от региона')
plt.xlabel('Регион')
plt.ylabel('mpg')
plt.show()


# In[ ]:





# ### 3.2 Нарисуйте ящики с усами, показывающие распределение mpg, в зависимости от количества цилиндров автомобиля

# In[19]:


sns.boxplot(x='cylinders', y='mpg', data=data)
plt.title('Распределение mpg в зависимости от количества цилиндров автомобиля')
plt.xlabel('Kоличествo цилиндров')
plt.ylabel('mpg')
plt.show()


# ## Задача 4

# ### 4.1 Нарисуйте scatter plot количества лошадинных сил против рабочего объема (displacement). Что вы думаете об их связи между собой?

# In[21]:


sns.scatterplot(x='displacement', y='horsepower', data=data)
plt.title('Лошадиные силы vs Рабочий объем')
plt.xlabel('Рабочий объем (displacement)')
plt.ylabel('Лошадиные силы (horsepower)')
plt.show()


# In[22]:


correlation_coefficient = data['horsepower'].corr(data['displacement'])


# In[23]:


correlation_coefficient


# In[ ]:


#Чем больше объем двигателя, тем больше лошадиных сил у автомобиля, и наоборот (положительная линейная зависимость)


# ### 4.2 Нарисуйте scatter plot  рабочего объема против mpg. Что вы думаете об их связи между собой?

# In[24]:


sns.scatterplot(x='displacement', y='mpg', data=data)
plt.title('MPG vs Рабочий объем')
plt.xlabel('Рабочий объем (displacement)')
plt.ylabel('MPG')
plt.show()


# In[26]:


correlation_coefficient2 = data['mpg'].corr(data['displacement'])


# In[27]:


correlation_coefficient2


# In[28]:


#Сильная отрицательная линейная зависимость.
#Это вполне логично, так как большой объем двигателя часто связан с более высоким расходом топлива.


# ### 4.3 Нарисуйте scatter plot  рабочего объема против mpg и покрасьте точки в зависимости от страны производства автомобиля (hue=...).  Что вы думаете?

# In[30]:


sns.scatterplot(x='displacement', y='mpg', hue='origin', data=data)
plt.title('MPG vs Рабочий объем')
plt.xlabel('Рабочий объем (displacement)')
plt.ylabel('mpg')
plt.legend(title='Производство', loc='upper right')
plt.show()


# ## Часть 2 - Данные из бизнеса

# ## Задача 5

# Прочитайте данные ('listings.csv') в Pandas dataframe и покажите их голову. Это данные от AirBnB, описывающие часть квартир и комнат на сдачу в Амстердаме.

# In[31]:


df = pd.read_csv('listings.csv')


# In[32]:


df.head()


# ## Задача 6

# ### 6.1 Используя countplot, ответьте на вопрос: 
# ### Какой тип комнаты самый распространенный в Амстердаме? https://seaborn.pydata.org/generated/seaborn.countplot.html

# In[33]:


sns.countplot(x='room_type', data=df)
plt.title('Распределение типов комнат в Амстердаме')
plt.xlabel('Тип комнаты')
plt.ylabel('Количество')
plt.show()


# 6.2 Нарисуйте ящики с усами, показывающие распределение цены в зависимости от района, в котором находится квартира/комната. Видите ли вы проблему?

# In[34]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='neighbourhood', y='price', data=df)
plt.title('Распределение цен в зависимости от района в Амстердаме')
plt.xlabel('Район')
plt.ylabel('Цена')
plt.xticks(rotation=90)  
plt.show()


# In[ ]:


#большой диапазон между боксами  и аутлаерс, 
#что усложняет читаемость графика. можно убрать отображение аутлаерс, чтобы 
#анализировать основную структуру распределения цен,а насчет аутлаерс провести более глубокий анализ и вывести причины.  


# ### 6.3 Нарисуйте ящики с усами, показывающие распределение цены в зависимости от района, в котором находится квартира/комната, используйте следующую команду в boxplot showfliers=False. Что изменилось?

# In[36]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='neighbourhood', y='price', data=df, showfliers=False)  
plt.title('Распределение цен в зависимости от района в Амстердаме')
plt.xlabel('Район')
plt.ylabel('Цена')
plt.xticks(rotation=90)
plt.show()


# ### 6.4 Создайте гистограмму цен для самого распространенного типа комнат в Амстердаме. С какими проблемами вы столкнулись?

# In[40]:


most_common_room_type = df['room_type'].mode()[0]

filter_df = df[df['room_type'] == most_common_room_type]

plt.figure(figsize=(10, 6))
sns.histplot(filter_df['price'], bins=30, kde=True)
plt.title(f'Гистограмма цен для {most_common_room_type} в Амстердаме')
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.show()


# In[ ]:


#Есть аутлаерс, они искажают общую картину и делают гистограмму трудночитаемой.

