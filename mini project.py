#!/usr/bin/env python
# coding: utf-8

# # Рейсы в США с Pandas, Matplotlib и Seaborn

# ![image.png](attachment:image.png)

# ### Задача:
# - написать код и выполнить вычисления в ячейках ниже (там где будут вопросы)

# In[2]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# * Скачать данные [архив](https://drive.google.com/file/d/1lGEDDVgD8QYMf9Fio_NG6jBWOF4Ttr7e/view?usp=sharing) (В архиве ~ 111 Мб, в разархивированном виде – ~ 690 Мб). Не нужно распаковывать — Pandas может сам это сделать.
# * Поместите его в папку там где лежит ваш Jupyter.
# * Набор данных содержит информацию о перевозчиках и рейсах между аэропортами США в течение 2008 года.
# * Описание столбцов доступно [здесь] (https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ). Посетите этот сайт, чтобы найти значение кодов отмены рейса. И в целом лучше понимать что за данные у нас есть.

# Рассмотрим следующие термины, которые мы используем:
# * уникальный рейс – запись (строка) в наборе данных
# * завершенный рейс – рейс, который не отменен (Cancelled==0 в наборе данных)
# * код рейса – комбинация ['UniqueCarrier','FlightNum'], т.е. UA52
# * код аэропорта – трехбуквенный псевдоним аэропорта из столбцов «Origin» или «Dest».

# **Чтение данных и создание объекта Pandas ``DataFrame``**
# 

# In[3]:


dtype = {
    "DayOfWeek": np.uint8,
    "DayofMonth": np.uint8,
    "Month": np.uint8,
    "Cancelled": np.uint8,
    "Year": np.uint16,
    "FlightNum": np.uint16,
    "Distance": np.uint16,
    "UniqueCarrier": str,
    "CancellationCode": str,
    "Origin": str,
    "Dest": str,
    "ArrDelay": np.float16,
    "DepDelay": np.float16,
    "CarrierDelay": np.float16,
    "WeatherDelay": np.float16,
    "NASDelay": np.float16,
    "SecurityDelay": np.float16,
    "LateAircraftDelay": np.float16,
    "DepTime": np.float16,
}


# In[4]:


flights_df = pd.read_csv('flights_2008.csv.bz2', usecols=dtype.keys())


# **Проверьте количество строк и столбцов и распечатайте имена столбцов.**

# In[5]:


print(flights_df.shape)
print(flights_df.columns)


# **Распечатайте первые 5 строк набора данных.**

# In[5]:


flights_df.head()


# **Изучите типы данных всех столбцов.**

# In[75]:


flights_df.info()


# **Получите базовую статистику по каждому столбцу.**

# In[76]:


flights_df.describe()


# **Подсчитайте уникальных перевозчиков и определите их относительную долю рейсов:**

# In[13]:


flights_df["UniqueCarrier"].nunique()


# In[27]:


# Количество рейсов по перевозчикам 
flights_df.groupby("UniqueCarrier").size().plot(kind="bar");


# **Мы также можем группировать по категориям, чтобы рассчитывать различные агрегированные статистические данные.**
# 
# **Например, поиск топ-3 кодов рейсов с наибольшим общим расстоянием, пройденным в 2008 году.**

# In[21]:


flights_df.groupby(["UniqueCarrier", "FlightNum"])["Distance"].sum().sort_values(ascending=False).iloc[:3]


# **Другой способ:**

# In[23]:


flights_df.groupby(["UniqueCarrier", "FlightNum"]).agg({"Distance": [np.mean, np.sum, "count"], "Cancelled": np.sum}
                                                      ).sort_values(("Distance", "sum"), ascending=False).iloc[0:3]


# **Количество рейсов по дням недели и месяцам:**

# In[24]:


pd.crosstab(flights_df.Month, flights_df.DayOfWeek)


# **Гистограмма дальности полета:**

# In[28]:


flights_df.hist("Distance", bins=20);


# **Составление гистограммы частоты рейсов по дате.**

# In[29]:


flights_df["Date"] = pd.to_datetime(
    flights_df.rename(columns={"DayofMonth": "Day"})[["Year", "Month", "Day"]]
)


# In[22]:


flights_df.head()


# In[31]:


num_flights_by_date = flights_df.groupby("Date").size()


# In[32]:


num_flights_by_date.plot();


# **Видите ли вы какую-то зависимость по неделям, еженедельную динамику выше? И ниже?**

# In[33]:


num_flights_by_date.rolling(window=7).mean().plot();


# **Нам понадобится новый столбец в нашем наборе данных — час отправления, давайте создадим его.**
# 
# Как мы видим, `DepTime` распределяется от 1 до 2400 (оно задаётся в формате `hhmm`, проверьте [описание столбца](https://www.transtats.bts.gov/Fields.asp?Table_ID=236 ) снова). Мы будем рассматривать час отправления как `DepTime` // 100 (разделите на 100 и примените функцию `floor`). Однако теперь у нас будет и час 0, и час 24. Час 24 звучит странно, вместо этого мы установим его равным 0 (типичное несовершенство реальных данных, однако вы можете проверить, что оно влияет только на 521 строку, что вроде ничего страшного). Итак, теперь значения нового столбца `DepHour` будут распределены от 0 до 23. Есть некоторые недостающие значения, пока мы не будем их заполнять, а просто проигнорируем.

# In[20]:


flights_df["DepHour"] = flights_df["DepTime"] // 100
flights_df["DepHour"].replace(to_replace=24, value=0, inplace=True)


# In[21]:


flights_df["DepHour"].describe()


# **<font color='red'>Вопрос 1.</font> Набор данных включает как отмененные, так и завершенные рейсы. Определите, есть ли больше выполненных или отмененных рейсов, и посчитайте числовую разницу между ними.** <br>
# 
# - Отмененных рейсов больше, чем выполненных на 329.
# - Выполнено рейсов больше отмененных на 6 734 860.
# - Отмененные рейсы превышают выполненные на 671.
# - Выполнено рейсов больше, чем отменено на 11 088 967.

# In[7]:


completed_flights = (flights_df['Cancelled'] == 0).sum()
cancelled_flights = (flights_df['Cancelled'] == 1).sum()

difference = completed_flights - cancelled_flights
print(difference)


# In[ ]:


#Выполнено рейсов больше отмененных на 6 734 860.


# **<font color='red'>Вопрос 2.</font> Найдите рейс с наибольшей задержкой вылета и рейс с наибольшей задержкой прибытия. Летят ли они в один аэропорт, и если да, то какой у него код?**
# 
# - да, ATL
# - да, HNL
# - да, MSP
# - нет

# In[9]:


flights_df['ArrDelay'].max()


# In[10]:


flights_df['DepDelay'].max()


# In[23]:


flights_df.loc[flights_df['ArrDelay'].idxmax()]


# In[13]:


flights_df.loc[flights_df['DepDelay'].idxmax()]


# In[14]:


max_departure_delay = flights_df.loc[flights_df['DepDelay'].idxmax()]
max_arrival_delay = flights_df.loc[flights_df['ArrDelay'].idxmax()]


if max_departure_delay['Dest'] == max_arrival_delay['Dest']:
    print("\nОба рейса летят в один аэропорт.")
    print("Код аэропорта:", max_departure_delay['Dest'])
else:
    print("\nРейсы летят в разные аэропорты.")


# **<font color='red'>Вопрос 3.</font> Найдите перевозчика, у которого наибольшее количество отмененных рейсов.**
# 
# - АА
# - MQ
# - ВН
# - СО

# In[18]:


cancelled_flights = flights_df[flights_df['Cancelled'] == 1].groupby('UniqueCarrier')['Cancelled'].sum()


# In[19]:


cancelled_most = cancelled_flights.idxmax()
print(cancelled_most)


# **<font color='red'>Вопрос 4.</font> Давайте рассмотрим время отправления и рассмотрим распределение по часам (столбец `DepHour`, который мы создали ранее). В какой час самый высокий процент рейсов?**<br>
# 
# *Подсказка:* Проверьте формат времени [здесь](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ).
# 
# - 1 am 
# - 5 am  
# - 8 am 
# - 3 pm 

# In[36]:


total_flights_by_hour = flights_df['DepHour'].value_counts()

percentage = total_flights_by_hour / total_flights_by_hour.sum() * 100

hour_with_highest_percentage = percentage.idxmax()
highest_percentage = percentage.max()

print(f"Час с самым высоким процентом рейсов: {hour_with_highest_percentage}")
print(f"Процент рейсов в этот час: {highest_percentage:.2f}%")


# **<font color='red'>Вопрос 5.</font> Хорошо, теперь давайте рассмотрим распределение отмененных рейсов по времени. В какой час наименьший процент отмененных рейсов?**<br>
# 
# - 2 am 
# - 9 pm  
# - 8 am  
# - 3 am

# In[64]:


cancelled_flights_by_hour = flights_df[flights_df['Cancelled'] == 1].groupby('DepHour').size()

# Создание полного списка часов в пределах 0-23
all_hours = list(range(24))

# Добавление нулевых значений для часов, которые могут отсутствовать в данных
cancelled_flights_by_hour = cancelled_flights_by_hour.reindex(all_hours, fill_value=0)

percentage_cancelled_flights_by_hour = cancelled_flights_by_hour / cancelled_flights_by_hour.sum() * 100

hour_with_lowest_percentage_cancelled = percentage_cancelled_flights_by_hour.idxmin()
lowest_percentage_cancelled = percentage_cancelled_flights_by_hour.min()

print(f"Час с наименьшим процентом отмененных рейсов: {hour_with_lowest_percentage_cancelled}")
print(f"Процент отмененных рейсов в этот час: {lowest_percentage_cancelled:.2f}%")


# In[ ]:





# In[ ]:





# **<font color='red'>Вопрос 6.</font> Есть ли какой-нибудь час, в который вообще не было отмененных рейсов? Проверить все, что относится.**
# 
# - 3
# - 19
# - 22
# - 4

# In[72]:


flights_df.groupby('DepHour')['Cancelled'].sum()


# In[ ]:


#в 3 часа не было отмененных рейсов 


# **<font color='red'>Вопрос 7.</font> Найдите самый загруженный час или, другими словами, час, когда количество вылетающих рейсов достигает максимума.**<br>
# 
# *Подсказка:* Учитывайте только *завершенные* рейсы.
# 
# - 4
# - 7
# - 8
# - 17

# In[79]:


completed_flights = flights_df[flights_df['Cancelled'] == 0]

completed_flights_by_hour = completed_flights.groupby('DepHour')['DepHour'].count()

max_hour = completed_flights_by_hour.idxmax()
max_completed_flights = completed_flights_by_hour.max()

print(f"Самый загруженный час: {max_hour}")
print(f"Количество: {max_completed_flights}")


# In[78]:


completed_flights_by_hour


# In[ ]:





# In[ ]:





# **<font color='red'>Вопрос 8.</font> Поскольку мы знаем час отправления, было бы интересно изучить среднюю задержку для соответствующего часа. Бывают ли случаи, когда самолеты в среднем вылетали раньше положенного? И если да, то в какое время это произошло?**<br>
# 
# *Подсказка:* Учитывайте только *завершенные* рейсы.
# 
# - нет, таких случаев не бывает
# - да, в 5-6 утра
# - да, в 9-10 утра
# - да, в 14-16.00

# In[81]:


completed_flights = flights_df[flights_df['Cancelled'] == 0]


# In[84]:


average_delay_by_hour = completed_flights_df.groupby('DepHour')['DepDelay'].mean()


# In[86]:


average_delay_by_hour


# In[87]:


earliest_departure_hour = average_delay_by_hour.idxmin()
earliest_departure_delay = average_delay_by_hour.min()

print(f"Cамолеты вылетают раньше положенного в: {earliest_departure_hour}")
print(f"Средняя задержка вылета в этот час: {earliest_departure_delay:.2f} минут")


# In[ ]:





# **<font color='red'>Вопрос 9.</font> Учитывая только выполненные перевозчиком рейсы, которого вы нашли в вопросе 3, найдите распределение этих рейсов по часам. В какое время вылетает наибольшее количество самолетов?**<br>
# 
# - в полдень
# - в 7 утра
# - в 8 утра
# - в 10 утра

# In[96]:


cancelled_most_carrier = flights_df[flights_df['Cancelled'] == 1].groupby('UniqueCarrier')['Cancelled'].sum().idxmax()


completed_flights_carrier = flights_df[(flights_df['UniqueCarrier'] == cancelled_most_carrier) 
                                          & (flights_df['Cancelled'] == 0)]

completed_flights = completed_flights_carrier.groupby('DepHour').size()

busiest_hour_completed_flights = completed_flights.idxmax()
max_completed_flights = completed_flights.max()

print(f"Час, в который выполняется наибольшее количество рейсов: {busiest_hour_completed_flights}")
print(f"Количество выполненных рейсов в этот час: {max_completed_flights}")


# In[ ]:





# **<font color='red'>Вопрос 10.</font> Найдите топ-10 перевозчиков по количеству *выполненных* рейсов (столбец _UniqueCarrier_)?**
# 
# **Что из перечисленного ниже _нет_ в вашем списке топ-10?**
# - DL
# - AA
# - OO
# - EV 

# In[6]:


completed_flights = flights_df[flights_df['Cancelled'] == 0]

top_carriers = completed_flights.groupby('UniqueCarrier').size().nlargest(10)

print(top_carriers)


# In[ ]:


#из перечисленного ниже нет EV


# **<font color='red'>Вопрос 11.</font> Постройте распределение причин отмены рейса (CancellationCode).**
# 
# **Какая наиболее частая причина отмены рейса? (Используйте эту [ссылку](https://www.transtats.bts.gov/Fields.asp?Table_ID=236), чтобы перевести коды в причины)**
# - Перевозчик
# - Погодные условия
# - Национальная воздушная система
# - Причины безопасности

# In[8]:


cancellation_distribution = flights_df['CancellationCode'].value_counts()

plt.figure(figsize=(8, 6))
cancellation_distribution.plot(kind='bar', color='skyblue')
plt.title('Распределение причин отмены рейсов')
plt.xlabel('Причина отмены рейса (CancellationCode)')
plt.ylabel('Количество отмененных рейсов')
plt.show()

most_cancellation_reason = cancellation_distribution.idxmax()
print(f"Наиболее частая причина отмены рейса: {most_cancellation_reason}")


# In[ ]:





# **<font color='red'>Вопрос 12.</font> Какой маршрут наиболее частый по количеству рейсов?**
# 
# (Обратите внимание на столбцы _'Origin'_ и _'Dest'_. Рассматривайте направления _A->B_ и _B->A_ как _разные_ маршруты)
# 
#   - Нью-Йорк – Вашингтон (JFK-IAD)
#   - Сан-Франциско – Лос-Анджелес (SFO-LAX)
#   - Сан-Хосе – Даллас (SJC-DFW)
#   - Нью-Йорк – Сан-Франциско (JFK-SFO)

# In[9]:


flights_df['Route'] = flights_df['Origin'] + ' -> ' + flights_df['Dest']

most_common_route = flights_df['Route'].value_counts().idxmax()
num_flights_on_route = flights_df['Route'].value_counts().max()

print(f"Самый частый маршрут: {most_common_route}")
print(f"Количество рейсов: {num_flights_on_route}")


# **<font color='red'>Вопрос 13.</font> . Найдите топ-5 задержанных маршрутов (посчитайте, сколько раз они задерживались при отправлении). Из всех рейсов по этим 5 маршрутам посчитайте все рейсы, погодные условия которых способствовали задержке.**
# 
# _Подсказка_: учитывайте только положительные задержки
# 
# - 449
# - 539
# - 549
# - 668

# In[10]:


delayed_flights = flights_df[flights_df['DepDelay'] > 0]

top_delayed_routes = delayed_flights['Route'].value_counts().nlargest(5).index

top_delayed_routes_flights = flights_df[flights_df['Route'].isin(top_delayed_routes)]

weather_delayed_flights = top_delayed_routes_flights[top_delayed_routes_flights['WeatherDelay'] > 0]

print("Топ-5 задержанных маршрутов:")
print(top_delayed_routes)
print("\nРейсы с погодными условиями, способствующими задержке:")
print(weather_delayed_flights[['FlightNum', 'Route', 'WeatherDelay']])


# In[ ]:





# In[ ]:





# In[ ]:





# **<font color='red'>Вопрос 14.</font> В каком месяце происходит наибольшее количество отмен по вине Carrier?**
# 
# - Май
# - Январь
# - Сентябрь
# - Апрель

# In[30]:


carrier_cancellation_flights = flights_df[(flights_df['Cancelled'] == 1) & (flights_df['CancellationCode'] == 'A')]

carrier_cancellation_by_month = carrier_cancellation_flights.groupby('Month').size()

max_cancellation_month = carrier_cancellation_by_month.idxmax()

print(max_cancellation_month)


# In[ ]:





# **<font color='red'>Вопрос 15.</font> Определите перевозчика с наибольшим количеством отмен из-за перевозчика в соответствующем месяце из предыдущего вопроса.**
# 
# - 9E
# - EV
# - HA
# - AA

# In[36]:


carrier_cancellation_month = flights_df[(flights_df['Cancelled'] == 1) 
                                        & (flights_df['CancellationCode'] == 'A') 
                                        & (flights_df['Month'] == 4)]

max_cancellations_carrier_month = carrier_cancellation_month['UniqueCarrier'].value_counts().idxmax()

print(f"В месяце апрель перевозчик с наибольшим количеством отмен из-за перевозчика: {max_cancellations_carrier_month}")


# In[ ]:




