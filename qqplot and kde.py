#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[ ]:


#zadanie1


# In[7]:


np.random.seed(105)
list1 = np.random.choice(range(1, 7), size=1000)


# In[9]:


plt.hist(list1)
plt.show()


# In[15]:


plt.hist(list1, bins=np.arange(1, 8, 1), rwidth=0.9, density=True)
plt.title('Вероятности бросков игральной кости')
plt.xlabel('Значение на кости')
plt.ylabel('Вероятноя частота')
plt.xticks(range(1, 7))
plt.show()


# In[ ]:


#zadanie2


# In[44]:


np.random.seed(105) 
list2= np.random.normal(50,15,10000)


# In[45]:


mean = 50
std_dev = 15


# In[46]:


outliers = np.random.normal(200, 10, 10)
outl_list = np.concatenate([list2, outliers])


# In[47]:


outl_mean = np.mean(outl_list)
outl_std = np.std(outl_list)

print(outl_mean , outl_std)


# In[48]:


plt.hist(list2, bins=np.arange(0,105,5), density=True, edgecolor='black', alpha=0.7)
plt.axvline(outl_mean, color='red', linestyle='dashed', linewidth=2, label='Новое среднее')
plt.axvline(outl_mean + outl_std, color='green', linestyle='dashed', linewidth=2, label='Новое среднее + std')
plt.axvline(outl_mean - outl_std, color='green', linestyle='dashed', linewidth=2, label='Новое среднее - std')

plt.title('Модифицированное распределение с выбросами')
plt.xlabel('Значение')
plt.ylabel('Относительная частота')
plt.legend()
plt.show()

print(f"Исходное среднее: {mean}, Исходное стандартное отклонение: {std_dev}")
print(f"Новое среднее: {outl_mean}, Новое стандартное отклонение: {outl_std}")


# In[ ]:





# In[ ]:


#zadanie3


# In[50]:


np.random.seed(105)


# In[51]:


citya = np.random.normal(50000, 15000, 1000)
cityb = np.random.normal(55000, 20000, 1000)


# In[58]:


sns.kdeplot(citya, label='City A')
sns.kdeplot(cityb, label='City B')

plt.title('Распределение доходов в City A и City B')
plt.xlabel('Доход')
plt.ylabel('Плотность')

plt.axvline(np.mean(citya), color='blue', linestyle='--', linewidth=2, label='Средний доход City A')
plt.axvline(np.mean(cityb), color='orange', linestyle='--', linewidth=2, label='Средний доход City B')

plt.legend()
plt.show()


# In[ ]:





# In[ ]:


#zadanie4


# In[59]:


np.random.seed(105)
time1 = np.random.exponential(10,1000)


# In[61]:


plt.hist(time1)
plt.show()


# In[63]:


time2 = np.random.choice(time1, size=100, replace=False)

plt.figure(figsize=(10, 6))

sns.kdeplot(time2, label='Выборка (KDE)')

plt.hist(time2, bins=20, density=True, alpha=0.5, edgecolor='black', label='Выборка (гистограмма)')

plt.title('Распределение времени обслуживания клиентов в выборке')
plt.xlabel('Время обслуживания')
plt.ylabel('Плотность')
plt.legend()
plt.show()


# In[64]:


import statsmodels.api as sm


sm.qqplot(np.array(time2), line='r')
plt.show()


# In[ ]:




