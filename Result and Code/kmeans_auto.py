from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import nltk
from nltk.corpus import stopwords


files = ['Fifa.txt','USelections.txt','SuperTuesday.txt']
files1 = ['Fifa_1.txt','USelections_1.txt','SuperTuesday_1.txt']
#Total tweets = 6123 = 1902 + 2200 + 2021
a = []
for i in files:
	fo = open(i ,'r')
	data = fo.readlines()
	for j in data:
		a.append(j)

print len(a)


tf = TfidfVectorizer(analyzer='word', min_df = 0,  encoding = 'utf-8', decode_error = 'replace',stop_words = 'english')

tfidf_matrix =  tf.fit_transform(a)
feature_names = tf.get_feature_names() 
print len(feature_names)
print tfidf_matrix
print "clustering"
km = KMeans(n_clusters=3, init='k-means++', max_iter=5000, n_init=1)

km.fit_predict(tfidf_matrix,y=None)

print km.labels_
print len(km.labels_)
a = [0,0,0]
for i in range(1902):
	a[km.labels_[i]] +=1
print a

a = [0,0,0]
for i in range(1902,1902+2200):
	a[km.labels_[i]] +=1
print a

a = [0,0,0]
for i in range(1902+2200,6123):
	a[km.labels_[i]] +=1
print a







