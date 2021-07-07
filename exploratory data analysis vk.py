#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install wordcloud
get_ipython().system('pip install pymorphy2')


# In[295]:


import pandas as pd
import numpy as np
import ast
from numpy import nan
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import tqdm
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[296]:


group_data = pd.read_excel(r'C:\Users\User\Desktop\Прога\Питон\аналитика вк\парсинг вк\group_data.xlsx')


# In[297]:


group_data['likes'] = group_data['likes'].apply(lambda x: ast.literal_eval(x)['count'])
group_data['post_source'] = group_data['post_source'].apply(lambda x: ast.literal_eval(x)['type'])
group_data['comments'] = group_data['comments'].apply(lambda x: ast.literal_eval(x)['count'])
group_data['reposts'] = group_data['reposts'].apply(lambda x: ast.literal_eval(x)['count'])


# In[298]:


group_data = group_data.drop(columns=['Unnamed: 0', 'from_id', 'group_id', 'copy_history', 'signer_id', 'attachments'])


# In[299]:


group_ids = [-65596623, -57846937, -150550417, -28905875, -73247559, -66678575, -31480508, -88245281, -29573241]
group_names = ['FTP', 'MDK', 'Reddit', 'Рифмы и панчи', 'ВПШ', 'Овсянка', 'Пикабу', 'На приеме у шевцова', 'NR']
group_names_df = pd.DataFrame({'owner_id':group_ids, 'group_names':group_names})


# In[300]:


group_data = group_data.merge(group_names_df, on='owner_id', how='left')


# In[301]:


group_data['date'] = [datetime.fromtimestamp(x) for x in group_data['date']]
group_data['day'] = group_data['date'].dt.date
group_data['hour'] = group_data['date'].dt.hour
group_data['day_of_week'] = group_data['date'].dt.weekday


# Обработаем текст описания

# In[302]:


def preprocessing_text(texts):
    my_words = set({'это', "который", "комментарий", "новый", "год"})
    eng = set(stopwords.words('english'))
    rus = set(stopwords.words('russian'))
    stop = eng | rus | my_words
    
    regex = re.compile('[^а-яА-Я]')
    preprocess_texts = []
    for i in  tqdm.tqdm(range(len(texts))):
        text = texts[i].lower()
        text = regex.sub(' ', text)
        word_tokens = word_tokenize(text)
        filtered_str = [w for w in word_tokens if not w in stop] 
        preprocess_texts.append( ' '.join(filtered_str))
    
    return preprocess_texts
        


# In[303]:


def text_stem(texts):
    st = SnowballStemmer("russian")
    stem_text = []
    for text in tqdm.tqdm(texts):
        tokenised_text = word_tokenize(text)
        stem_text.append(' '.join([st.stem(word) for word in tokenised_text]))
    return stem_text


# In[304]:


def text_lem(texts):
    lem = pymorphy2.MorphAnalyzer()
    lem_text = []
    for text in tqdm.tqdm(texts):
        tokenised_text = word_tokenize(text)
        lem_text.append(' '.join([lem.parse(word)[0].normal_form for word in tokenised_text]))
    return lem_text


# In[305]:


group_data.iloc[8414, :]


# In[306]:


group_data['text'].fillna('', inplace=True)
group_data['text'] = preprocessing_text(group_data['text'])
group_data['text'] = text_lem(group_data['text'])
group_data['text'] = preprocessing_text(group_data['text'])


# In[307]:


text = " ".join(desc for desc in group_data.text)
wordcloud = WordCloud(background_color="white").generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# word cloud отдельно по пабликам

# In[350]:


df = group_data.groupby('group_names')['text'].agg(lambda x: ' '.join(x))


# In[362]:


for group, text in df.items():
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('group name: {}'.format(group))
    plt.show()


# In[338]:


group_data.groupby('group_names', as_index=False)['text']


# In[308]:


group_data.head()


# Удалим выбросы по 3 сигме и заполним пропуски в данных

# In[309]:


group_data.fillna(0, inplace=True)
group_data = group_data[group_data['post_source'] != 'mvk']
group_data = group_data[(np.abs(stats.zscore(group_data[['comments', 'likes', 'reposts']])) < 3).all(axis=1)]


# In[310]:


group_data.groupby('post_source')['likes', 'reposts', 'comments'].agg(['count', 'mean', 'median'])


# In[366]:


group_data.groupby('group_names')['likes'].agg(['count', 'mean', 'median']).sort_values(by='mean', ascending=False)


# In[311]:


g = sns.FacetGrid(group_data, col="post_source");
g.map(sns.boxplot, 'likes');


# In[312]:


g = sns.FacetGrid(group_data, col="post_source");
g.map(sns.boxplot, 'reposts');


# In[313]:


g = sns.FacetGrid(group_data, col="post_source");
g.map(sns.boxplot, 'comments');


# Можем заметить, что у записей, опубликованных через api показатели выше. Полагаю, что паблики с большим количеством подписчиков заливают посты через api

# In[314]:


group_data.groupby(['post_source', 'group_names'])['likes'].agg(['count', 'mean', 'median']).sort_values(by='mean', ascending=False)


# Так и есть, скорее всего все портит паблик ВПШ, который в основном заливает посты через api, лайки на которых значительно выше, построим графики без него

# In[315]:


g = sns.FacetGrid(group_data[group_data['group_names'] != 'ВПШ'], col="post_source");
g.map(sns.boxplot, 'likes');


# В графиках мало что поменялось, проверим гипотезу о равенстве средних на уровне значимости 5%.
# Наблюдений много, поэтому воспользуемся асимптотикой

# In[316]:


vk = group_data[(group_data['post_source'] == 'vk') & (group_data['group_names'] != 'ВПШ')]['likes']
api = group_data[(group_data['post_source'] == 'api') & (group_data['group_names'] != 'ВПШ')]['likes']


# In[317]:


alpha = 0.05
nu1 = vk.mean()
nu2 = api.mean()
diff = nu2 - nu1
std1 = vk.var(ddof=1)
std2 = api.var(ddof=1)
z_obs = diff * np.sqrt(std1/vk.size + std2/api.size)
z_crit = stats.norm().ppf(1-alpha/2)
p_value = 2* (1 - (stats.norm().cdf(z_obs)))
print('z_obs:{}, z_crit:{}, p_value:{}'.format(z_obs, z_crit, p_value))


# Найдем самое лучшее время для публикации

# In[318]:


group_data.head()


# In[430]:


ad_by_h = group_data.groupby(['marked_as_ads', 'hour'])['likes'].agg('count')
state_pcts = ad_by_h.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
state_pcts1 = pd.DataFrame(state_pcts).reset_index()

state_pcts


# In[432]:


g = sns.FacetGrid(state_pcts1, col="marked_as_ads", aspect=3);
g.map(sns.barplot, 'hour', 'likes');


# In[327]:


sns.relplot(data=group_data, x='hour', y='likes', kind='line')


# In[328]:


sns.boxplot(data=group_data, x='hour', y='likes')


# In[329]:


sns.boxplot(data=group_data, x='day_of_week', y='likes')


# In[330]:


sns.boxplot(data=group_data, x='day_of_week', y='likes')


# In[319]:


sns.relplot(data=group_data, x="hour", y="likes", col="post_source", kind='line')


# In[391]:


g = sns.FacetGrid(group_data, col="group_names", col_wrap=3);
g.map(sns.boxplot, 'likes');


# In[392]:


g = sns.FacetGrid(group_data, col="group_names", col_wrap=3);
g.map(sns.histplot, 'likes', kde=True);


# In[378]:


group_data['word_num_desc'] = group_data['text'].apply(lambda x: len(x.split()))


# In[381]:


group_data.word_num_desc.describe()


# In[382]:


sns.heatmap(group_data[['likes', 'comments', 'reposts', 'word_num_desc']].corr(), annot=True)


# In[ ]:


group_da

