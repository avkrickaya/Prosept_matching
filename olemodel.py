#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[3]:


# model.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import sys



def matching(product_name):
    
    # Функция для добавления пропущенных пробелов в наименованиях.
    def add_spaces(text):
        spaced_text = re.sub(r'(?<=[a-zA-Z])(?=[а-яА-ЯёЁ])|(?<=[а-яА-ЯёЁ])(?=[a-zA-Z])', ' ', text)
        spaced_text = re.sub(r'(\S)\*(\S)', r'\1 * \2', spaced_text)
        spaced_text = re.sub(r'(\d+)([а-яА-ЯёЁa-zA-Z]+)', r'\1 \2', spaced_text)
        return spaced_text
    
        # Функция очистки,  токенизации, лемматизации и удаление стоп-слов.
    def preprocess_text(text):
        # Очистка текста
        cleaned_text = re.sub(r"[\(\),/.]", ' ', text) # r"[^a-zA-Zа-яА-ЯёЁ ]"

        # Токенизация
        tokens = word_tokenize(cleaned_text.lower())

        # Возвращение предобработанного текста
        return ' '.join(tokens)



    def get_rec_labse(product_name):
        preprocessed_name = preprocess_text(product_name)

        dealer_product_embedding = model_labse.encode([preprocessed_name])

        # Поиск наиболее похожих названий
        cosine_scores = util.pytorch_cos_sim(dealer_product_embedding, corpus_embeddings_labse)[0]

        top_matches_indices = cosine_scores.argsort(descending=True)[:5]
    #     top_matches_names = [names_corpus[i] for i in top_matches_indices]
        recommendations = data_products.loc[list(top_matches_indices)]['name'].tolist()

        return recommendations

    
    data_products = pd.read_csv('marketing_product.csv', sep=';')


    # удалили пропуск в названии 2 шт
    data_products.dropna(subset=['name'], inplace=True)
    # удалили одно название пустое
    empty_name_rows = data_products[data_products['name'].str.contains(r'^\s*$')]
    data_products.drop(index=empty_name_rows.index, inplace=True)


    #  добавить пропущенные пробелы в наименования товаров в столбцах 'name', 'name_1c', 'ozon_name' и 'wb_name'
    columns_to_apply = ['name', 'name_1c', 'ozon_name', 'wb_name']
    data_products[columns_to_apply] = data_products[columns_to_apply].astype(str).map(add_spaces)


    # обнаружено, что есть лишний пробел в слове Средство "C редство"
    # вручную исправим ошибку
    data_products['name_1c'] = data_products['name_1c'].str.replace('C редство', 'Средство')
    data_products['name'] = data_products['name'].str.replace('C редство', 'Средство')

    # в наименования оказались еще места, которые не были обработаны функцией и уберем эти проблемы вручную
    data_products['name'] = data_products['name'].str.replace('БЕТОНКОНТАКТготовый', 'БЕТОНКОНТАКТ готовый')
    data_products['name'] = data_products['name'].str.replace('"к', '" к')
    data_products['name'] = data_products['name'].str.replace('яблокаконцентрированное', 'яблока концентрированное')
    data_products['name'] = data_products['name'].str.replace('(сухой остаток 20%)', ' (сухой остаток 20%) ')
    data_products['name'] = data_products['name'].str.replace('.C', '. C')

    # пропуски в столбце name_1c заполнить значениями в сттолбце name 
    data_products['name_1c'].fillna(data_products['name'], inplace=True)
    
    data_products = data_products.dropna( subset=['name']).reset_index(drop=True)

    model_labse = SentenceTransformer('sentence-transformers/LaBSE')

    # очистка названий товаров заказчика в data_products
    names_corpus = data_products['name_1c'].apply(lambda x: preprocess_text(x)).tolist()
    
    corpus_embeddings_labse = model_labse.encode(names_corpus)
    
    recommendations = get_rec_labse(product_name)
    
    return recommendations

# тест
# product_name = 'Спрей для очистки полков в банях и саунах Universal Wood, 0,5л'

if __name__ == "__main__":
    # Check if an argument is passed
    if len(sys.argv) > 1:
        product_name = sys.argv[1]
        recommendations = matching(product_name)
        print(recommendations)
    else:
        print("No product name provided.")


matching(product_name)

