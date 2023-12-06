import pandas as pd
import numpy as np
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
tqdm.pandas() # для работы progress_apply в пандас

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from pymorphy3 import MorphAnalyzer

from sklearn.neighbors import NearestNeighbors



def matching(products_json:list, dealer_json:list):

    if isinstance(products_json, list):
        data_products = pd.DataFrame(products_json)
    if isinstance(dealer_json, list):
        data_dealers = pd.DataFrame(dealer_json)

    # подготовка датасетов
    # удалить продукты дубликаты, спарсенные в разные дни
    data_dealers = data_dealers[~data_dealers.drop(['id', 'date'], axis=1).duplicated(keep='first')]
    #Переменная для количества KNN
    k_matches = 5
    # удалить пропуски в столбце с именем
    data_products = data_products.dropna(subset=['name'])
    # заменить вручную ошибки, которые не обработаются функцией
    data_products['name'] = data_products['name'].str.replace('C редство', 'Средство')
    data_products['name'] = data_products['name'].str.replace('БЕТОНКОНТАКТготовый', 'БЕТОНКОНТАКТ готовый')
    data_products['name'] = data_products['name'].str.replace('"к', '" к')
    data_products['name'] = data_products['name'].str.replace('яблокаконцентрированное', 'яблока концентрированное')
    data_products['name'] = data_products['name'].str.replace('(сухой остаток 20%)', ' (сухой остаток 20%) ')
    data_products['name'] = data_products['name'].str.replace('.C', '. C')

    def preprocessing(data):
         # Добавляет пробелы 
         cleaned_text = re.sub(r'(?<=[a-zA-Z])(?=[а-яА-ЯёЁ])|(?<=[а-яА-ЯёЁ])(?=[a-zA-Z])', ' ', data)
         cleaned_text = re.sub(r'(\S)\*(\S)', r'\1 * \2', cleaned_text)
         cleaned_text = re.sub(r'(\d+)([а-яА-ЯёЁa-zA-Z]+)', r'\1 \2', cleaned_text)

            # Функция очистки,  токенизации, лемматизации и удаление стоп-слов.
         # Очистка текста
         cleaned_text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ 0-9,.:]", ' ', cleaned_text)
         cleaned_text = re.sub(r"[,]", ' ', cleaned_text)

         # Токенизация
         tokens = word_tokenize(cleaned_text.lower())

         morph = MorphAnalyzer()
         lemmas = [morph.parse(word)[0][2] for word in tokens]

         # Удаление стоп-слов
         stop_words = set(stopwords.words('russian') + stopwords.words('english'))
         filtered_words = [lemma for lemma in lemmas if lemma not in stop_words]

         # Возвращение предобработанного текста
         return ' '.join(filtered_words)

    # обработка текста
    data_products['name_clean'] = (data_products['name'].apply(
                                   lambda x: preprocessing(x))
                                   )
    data_dealers['product_name_clean'] = (data_dealers['product_name'].apply(
                                          lambda x: preprocessing(x))
                                          )

    # Эмбендинги продуктов
    vectorizer = TfidfVectorizer(min_df=1)
    tf_idf_product = vectorizer.fit_transform(list(data_products['name_clean']))

    # обучение модели k_maches ближайших соседей по коминусу
    nbrs = NearestNeighbors(n_neighbors=k_matches, 
                            n_jobs=-1, 
                            metric="cosine").fit(tf_idf_product)

    # эмбендинги диллеров и поиск k_maches ближайших соседей, их `дистанцию` и индексы
    tf_idf_dealers = vectorizer.transform(list(data_dealers['product_name_clean']))
    distances, dealers_indices = nbrs.kneighbors(tf_idf_dealers)
    
    # Пустые списки для заполнения датафрейма
    dealers_name_key= []
    product_id_list = []
    
    # цикл проходящий по всем найденным соседям
    for i, dealers_index in enumerate(dealers_indices):
        # фиксируем key продукта диллера
        dealers_name_key.append(list(data_dealers['product_key'])[i])
        # формируем список из всех выбранных моделью названий product
        product_id = [list(data_products['id'])[index] for index in dealers_index]
        
        # фиксируем `дистанции`
        confidence = [1 - round(dist, 2) for dist in distances[i]]
        # находим среднее, с помощью него отбросим все варианты меньше среднего квадратичного
        mean_con = np.mean(confidence)
        for con in range(k_matches):
            # если confidence[con]  меньше, обрубим список по данное значение и прервем цикл
            if confidence[con] < mean_con:
                product_id = product_id[:con]
                break
        
        # фиксируем key
        product_id_list.append(product_id)

    #код для создания DataFrame
    # Convert to df
    #df_orig_name = pd.DataFrame({'dealers_name_id':dealers_name_id})

    #df_lookups = pd.DataFrame({"product_key_list":product_key_list})
    #df_confidence = pd.DataFrame({"confidence":confidence_list})

    # bind columns
    #matches = pd.concat([df_orig_name, df_lookups, df_confidence], axis=1)

    # reorder columns | can be skipped
    #lookup_cols = list(matches.columns.values)
    #lookup_cols_reordered = [lookup_cols[0]]
    #for i in range(1, k_matches + 1):
    #    lookup_cols_reordered.append(lookup_cols[i])
    #    lookup_cols_reordered.append(lookup_cols[i + k_matches])
    #    lookup_cols_reordered.append(lookup_cols[i + 2 * k_matches])
    #matches = matches[lookup_cols_reordered]
    #return matches
    
    
    # создаем список словарей для хранения результатов
    result_list = []
    print(product_id_list)
        # итерируемся по каждому дилеру
    for index_key in range(0,len(dealers_name_key)):
        key = dealers_name_key[index_key]
        #print('faf',key)
            # итерируемся по топовым индексам
        for product_index in product_id_list[index_key]:
                # добавляем пару в список
            #print(key)
            answ = {'product_key': key, 'product_id': int(product_index)}  
            result_list.append(answ)

    return result_list