import re
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords

class preprocessing_prosept():
    
    # data-данные, name-колонка с названиями, key-ключи продуктов, data_with_product-маркер для обработки особенных ячеек
    def __init__(self, data, name, key, data_with_product=False):
        self.name = name
        self.key = key
        self.data = data.dropna(subset=[self.name])
        self.data_with_product = data_with_product
        

    def transform(self):
    
        # Обработка особенных названий только для данных с осоциациями
        if self.data_with_product == True:
            self.data[self.name] = self.data[self.name].str.replace('C редство', 'Средство')
            self.data[self.name] = self.data[self.name].str.replace('БЕТОНКОНТАКТготовый', 'БЕТОНКОНТАКТ готовый')
            self.data[self.name] = self.data[self.name].str.replace('"к', '" к')
            self.data[self.name] = self.data[self.name].str.replace('яблокаконцентрированное', 'яблока концентрированное')
            self.data[self.name] = self.data[self.name].str.replace('(сухой остаток 20%)', ' (сухой остаток 20%) ')
            self.data[self.name] = self.data[self.name].str.replace('.C', '. C')
            
        # Удаление дубликатов в спарсенных данных
        elif self.data_with_product == False: 
            self.data = self.data[~self.data[[self.name,self.key]].duplicated(keep='last')].reset_index(drop=True)
        else:
            raise ValueError('data_with_product is bool type')
            
        def preprocess_text(text):
            cleaned_text = re.sub(r'(?<=[a-zA-Z])(?=[а-яА-ЯёЁ])|(?<=[а-яА-ЯёЁ])(?=[a-zA-Z])', ' ', text)
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
        
        self.data[self.name] = self.data[self.name].astype(str).apply(preprocess_text)
        return self.data[[self.name,self.key]]