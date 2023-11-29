# Сервиса для полуавтоматической разметки товаров
#### Команда №14:
- Анастасия Крицкая - предобработка данных, тестирование моделей
- Олеся Круглякова - подбор способа валидации, тестирование моделей, отчётность, оформление
- Иван Прошин - тестирование моделей, оформление скрипта

### Задача проекта

Разработка решения, которое частично автоматизирует процесс сопоставления товаров. Основная идея - предлагать несколько товаров заказчика, которые с наибольшей вероятностью соответствуют размечаемому товару дилера.  Реализуется в виде онлайн сервиса, открываемого в веб-браузере.

### Задача ML-специалистов

Разработать рекомендательную модель, которая будет предлагать несколько наиболее вероятных вариантов названия товара у заказчика к названию товара, используемомому на сайте дилера. 

### Содержание репозитория с ML-решением
- [`prosept-hack-notebook.ipynb`](https://github.com/avkrickaya/Prosept_matching/blob/main/prosept-hack-notebook.ipynb) - Ноутбук с анализом, предобработкой данных, обучением модели, валидацией 
-  [`model.py`](https://github.com/avkrickaya/Prosept_matching/blob/main/model.py): скрипт прогнозирования `model.py`
-  [`requirements.txt`](https://github.com/avkrickaya/Prosept_matching/blob/main/requirements.txt) - список требуемых зависимостей
-  `marketing_product.csv` - таблица продуктов заказчика



#### Стек технологий
`Python3` `Pandas` `Sklearn` `Numpy` `pymorphy2` `NLTK` `sentence_transformers`

#### Метрика качества
Для валидации используется метрика **accuracy@5** - рассчитывается как доля случаев, когда хотя бы одна из правильных меток присутствует в топ-5 предсказанных.

Далее будем уточнять метрику, скорее всего, будем предсказывать не определённое количество в топе, а товары выше определённого порога вероятности.

### Инструкция по запуску скрипта 

Клонировать репозиторий

```git clone git@github.com:avkrickaya/Prosept_matching.git```   

Установить виртуальное окружение

```python3 -m venv venv```

Запустить виртуальное окружение

```source venv/bin/activate```

Установить все зависимости

```pip install -r requirements.txt```

Запустить скрипт

```python model.py "Ваше название продукта для подбора"```

Скрипт выдаст пять наиболее подходящих названий товаров из номенклатуры заказчика.

## Описание ML-решения 
Для решения задачи заказчика используется ранжирования, аллгоритм подбирает 5 наиболее вероятных названий.   
Выбран подход, в котором создаются эмбединги названий и далее подбор проходит с помощью косинусной близости. Протестировано несколько моделей для создания векторного представления.    
На текущей стадии работы для метчинга мы будем использовать только названия продуктов. Остальные признаки планируем задействоваь позже. Названия предварительно очищены от части символов, приведены к нижнему регисту, добавлены нехватающие пробелы.

- В качестве  baseline  модель: `TfidfVectorizer + косинусное сходство из sklearn`

- Лучший результат показала предобученная модель `LaBSE` из `sentence_transformers` + косинусное сходство+ `util.pytorch_cos_sim`.  
accuracy@5 = 91% на всей доступной выборке




## Взаимодействие 
Взаимодействие в приложении будет организовано по схеме  Batch deployment:    
Предсказания модели рассчитываются с  периодичностью -  ежедневно после окончания парсинга. Результат сохраняется в базу данных и предоставляется по запросу пользователя:
![image](https://github.com/avkrickaya/Prosept_matching/assets/139965241/54661a6b-f4c9-4133-9224-dd1c981fb5d4)


 Субъект | Действие
----------|----
Backend | Инициируется запуск ML-модуля по расписанию или сразу после окончания ежедневного парсинга|
ML + Backend | ML-модуль взаимодействует с бэкендом для получения данных таблиц marketing_product.csv и 'marketing_dealer.csv' из базы данных |
ML | ML-модуль проводит предобработку данных из таблиц и генерирует для каждой записи 5 рекомендованных названий продуктов |
Backend | Сгенерированные данные сохраняются в базе данных |
Frontend | Пользователь кликает на название продукта с целью сделать разметку, данные приходят из базы данных|


<br>
<br>


Ранее мы планировали организоввать взаимодействие Real-time, но из-за необходимости обрабатывать данные из marketing_product.csv для создания корпуса с именами для подбора время работы модуля критично увеличивается. Сейчас скрипт принимает одно название и выдаёт к нему 5 рекомендаций. Планируем переделать - на вход принимает таблицу с с спарсенными данными и выдаёт массив с рекомендациями в виде id продукта.


 Часть | Действие
----------|----
Frontend | Пользователь кликает на название продута с целью сделать разметку|
Backend + Backend| Backend инициирует запуск ML-модуля |
ML | ML-модуль взаимодействует с бэкендом для получения данных таблицы marketing_product.csv из базы данных |
ML | Модуль генерирует 5 рекомендованных названий продуктов|
Backend | рекомендации отправляются в Frontend |
Frontend | Пользователь видит 5 полученных рекомендаций и использует их для разметки |
