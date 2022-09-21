"""
抄袭自动检测分析的工具
"""
import os
import pickle
import numpy as np
import pandas as pd
import jieba
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from icecream import ic

# 加载停用词
with open('chinese_stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]
# ic(stopwords)

# 数据加载
news = pd.read_csv('sqlResult.csv', encoding='gb18030')
# ic(news.shape)
# ic(news.head())

# 处理缺失值
# ic(news[news.content.isna()].head())
news = news.dropna(subset=['content'])
# ic(news.shape)


# 分词
def split_text(text: str):
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    text2 = jieba.cut(text.strip())
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result

# ic(news.iloc[0].content)
# ic(split_text(news.iloc[0].content))

# 因为文件太大所以保存一下便于后续使用
if not os.path.exists('corpus.pkl'):
    corpus = list(map(split_text, [str(i) for i in news.content]))
    print(corpus[0])
    print(len(corpus))
    print(corpus[1])
    with open('corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file)
else:
    # 调用上次处理的结果
    with open('corpus.pkl', 'rb') as file:
        corpus = pickle.load(file)
        # print(corpus[0])
        # print(len(corpus))
        # print(corpus[1])

# 计算corpus的TF-IDF矩阵
countvectorizer = CountVectorizer(encoding='gb18030', min_df=0.015)
tfidftranformer = TfidfTransformer()
countvector = countvectorizer.fit_transform(corpus)
tfidf = tfidftranformer.fit_transform(countvector)
ic(tfidf.shape)

# 是否是自己的新闻
label = list(map(lambda source: 1 if '新华' in str(source) else 0, news.source))

# 数据切分
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label,
                                                    test_size=0.3, random_state=33)
# 分类器
clf = MultinomialNB()
# 分类器用的是 fit 和 predict
clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
# 要预测全部的
prediction = clf.predict(tfidf.toarray())
labels = np.array(label)

compare_news_index = pd.DataFrame({'prediction': prediction, 'labels': labels})
# 计算所有可疑文章的index
copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index
# 计算所有新华社的index
xinhuashe_news_index = compare_news_index[(compare_news_index['labels'] == 1)].index
print('可疑文章数：', len(copy_news_index))

normalizer = Normalizer()
scaled_array = normalizer.fit_transform(tfidf.toarray())

if not os.path.exists('label.pkl'):
    # 使用KMeans，对文章进行聚类
    kmeans = KMeans(n_clusters=25)
    k_labels = kmeans.fit_predict(scaled_array)
    with open('label.pkl', 'wb') as file:
        pickle.dump(k_labels, file)
    print('k_labels.shape', k_labels.shape)
else:
    with open('label.pkl', 'rb') as file:
        k_labels = pickle.load(file)


if not os.path.exists('id_class.pkl'):
    # 创建id_class
    id_class = {index: class_ for index, class_ in enumerate(k_labels)}
    with open('id_class.pkl', 'wb') as file:
        pickle.dump(id_class, file)
else:
    with open('id_class.pkl', 'rb') as file:
        id_class = pickle.load(file)


if not os.path.exists('class_id.pkl'):
    # 创建class_id
    class_id = defaultdict(set)
    for index, class_ in id_class.items():
        # 值统计新华社发布的class_id
        if index in xinhuashe_news_index.tolist():
            class_id[class_].add(index)

    with open('class_id.pkl', 'wb') as file:
        pickle.dump(class_id, file)
else:
    with open('class_id.pkl', 'rb') as file:
        class_id = pickle.load(file)

# 想找相似文本
def find_similar_text(cpindex, top=10):
    # 只在新华社发布的文章中进行查找
    dist_dict = {i: cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    # 从大到小进行排序
    return sorted(dist_dict.items(), key=lambda x: x[1][0], reverse=True)[:top]

cpindex = 3352
similar_list = find_similar_text(cpindex)
ic(similar_list)
print('怀疑抄袭:\n', news.iloc[cpindex].content)

# 找一篇相似的原文
similar2 = similar_list[0][0]
print('相似原文:\n', news.iloc[similar2].content)












