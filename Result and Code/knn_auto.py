import nltk
import numpy
import sklearn
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def get_data():
	files = ['Fifa.txt','USelections.txt','SuperTuesday.txt']
	files1 = ['Fifa_1.txt','USelections_1.txt','SuperTuesday_1.txt']
	index1 = [0,1,2]
	a = []
	for i in range(len(files1)):
		fo = open(files1[i] ,'r')
		data = fo.readlines()
		for j in data:
			data1 = [j,index1[i]]
			a.append(data1)

	print len(a)
	return a


from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
print "Getting data"
data1 = []
index = []
data = get_data()
for i in data:
	data1.append(i[0])
	index.append(i[1])

"""

for i in range(len(data1)):
	a = ""
	for j in data1[i]:
		a = ""
"""

#splitting 80:20
print "Splitting"
test_len = 20*6123/100
print test_len
te = []
while len(te) != test_len:
	a = random.randint(0,6123)
	if a not in te:
		te.append(a)
print "len te =",
print len(te)

test = []
test_index= []
train = []
train_index = []

for i in range(len(data1)):
	if i in te:
		test.append(data1[i])
		test_index.append(index[i])
	else:
		train.append(data1[i])
		train_index.append(index[i])

print len(test),len (test_index)
print len(train), len(train_index)



X = train
Y = train_index

#X = numpy.array(X)
#Y = numpy.array(Y)

#print X
#print Y

"""
vectorizer = TfidfVectorizer(min_df=10,
                            max_df = 0.2,
                             sublinear_tf=True,
                             use_idf=True)
"""
tf = TfidfVectorizer(analyzer='word', min_df = 0,  encoding = 'utf-8', decode_error = 'replace',stop_words = 'english')

X = tf.fit_transform(X)
k = 1

print "k=", k
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X,Y) 


testX = tf.transform(test)
#print testX
predict = neigh.predict(testX)
count = 0.0
for i in range(len(predict)):
	if predict[i] == test_index[i]:
		count  +=1.0
print count/test_len
