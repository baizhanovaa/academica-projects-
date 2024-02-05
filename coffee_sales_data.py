#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 


# In[3]:


df= pd.read_csv('coffee_sales_data.csv')


# In[29]:


df.info


# In[4]:


df.head()


# In[26]:


df_sales_weekday= df.groupby('День недели')['Количество Продаж'].sum().reset_index()


# In[27]:


df_sales_weekday


# In[28]:


plt.bar(df_sales_weekday['День недели'], df_sales_weekday['Количество Продаж'], color='green')

plt.xlabel('День недели')
plt.ylabel('Общее количество продаж')
plt.title('Распределение продаж кофе по дням недели')

plt.show()


# In[ ]:


#boxplot, который покажет распределение цен на разные типы кофе (Эспрессо, Латте, Капучино). 


# In[48]:


plt.figure(figsize=(10, 6))
df.boxplot(column='Цена за Чашку', by='Тип Кофе')

plt.xlabel('Тип Кофе')
plt.ylabel('Цена за Чашку')
plt.title('Распределение цен на разные типы кофе')

plt.show()


# In[ ]:





# In[ ]:


#Исследовать влияние выходных дней на продажи кофе в магазине с помощью столбчатой диаграммы,
#показывающей разницу в общем количестве продаж между выходными и будними днями. 


# In[52]:


sales_by_day = df.groupby('Выходной')['Количество Продаж'].sum().reset_index()


# In[53]:


sales_by_day


# In[55]:


plt.figure(figsize=(10, 6))
plt.bar(['Выходной', 'Будний день'], sales_by_day['Количество Продаж'], color=['blue', 'green'])

plt.xlabel('День недели')
plt.ylabel('Общее количество продаж')
plt.title('Разница в продажах между выходными и будними днями')

plt.show()


# In[ ]:




