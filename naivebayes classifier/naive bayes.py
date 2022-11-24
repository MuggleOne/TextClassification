import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# 读取正负例训练样本
with open(r'../datasets/negative_trainingSet.txt') as f:
    negative_trainingSet = f.readlines()

negative_labels = ["N"] * len(negative_trainingSet)
negative_trainingSet = {"title": negative_trainingSet, "label": ["N"] * len(negative_trainingSet)}
negative_df = pd.DataFrame(negative_trainingSet)

with open(r'../datasets/positive_trainingSet.txt') as f:
    positive_trainingSet = f.readlines()

positive_labels = ["Y"] * len(positive_trainingSet)
positive_trainingSet = {"title": positive_trainingSet, "label": positive_labels}
positive_df = pd.DataFrame(positive_trainingSet,index=None)

# 合并、打乱、保存
df = pd.concat([negative_df, positive_df], axis=0,ignore_index=True)

df = shuffle(df, n_samples=len(df))
df.to_csv(r"../datasets/trainingSet.csv",index=True)

#
features = list(df.iloc[:,0])
train_x = features[:int(len(features)*0.6)]
test_x = features[int(len(features)*0.6):]

labels =list(df.iloc[:,1])
train_label = labels[:int(len(labels)*0.6)]
test_label = labels[int(len(labels)*0.6):]

# tf-idf向量化表示
tf = TfidfVectorizer()
train_features = tf.fit_transform(train_x)
test_features = tf.transform(test_x)
tf.get_feature_names_out()

# 训练、预测
clf=MultinomialNB(alpha=0.001).fit(train_features,train_label)
pred_labels = clf.predict(test_features)
score=metrics.accuracy_score(pred_labels,test_label)
print(score)