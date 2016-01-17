import numpy
import lda
FIFA_File = file("Fifa_1.txt")
Election_File = file("USelections_1.txt")
Super_Tuesday_File = file("SuperTuesday_1.txt")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

X = []
fifa_tweets = 0


for line in FIFA_File:
	fifa_tweets+=1
	try:
		line = line.strip()
		X.append(line)
	except ValueError:
		pass

election_tweets = 0
for line in Election_File:
	election_tweets+=1
	try:
		line = line.strip() 
		X.append(line)
	except ValueError:
		pass

st_tweets = 0
for line in Super_Tuesday_File:
	st_tweets+=1
	try:
		line = line.strip()
		X.append(line)
	except ValueError:
		pass	

from nltk.corpus import stopwords
X = numpy.array(X)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

model = lda.LDA(n_topics=3, n_iter=3000, random_state=1)
model.fit(X)
probs = model.doc_topic_
fifa_list = [0,0,0]
election_list = [0,0,0]
st_list = [0,0,0]
probs = list(probs)
for i in xrange(0,fifa_tweets+1):
	pmax=list(probs[i]).index(max(probs[i]))
	fifa_list[pmax] +=1

for i in xrange(fifa_tweets+1,fifa_tweets+election_tweets):
	pmax=list(probs[i]).index(max(probs[i]))
	election_list[pmax] +=1

for i in xrange(fifa_tweets+1+election_tweets,fifa_tweets+election_tweets+st_tweets):
	pmax=list(probs[i]).index(max(probs[i]))
	st_list[pmax] +=1

print "FIFA: ",fifa_list
print "Elections: ",election_list
print "Super Tuesday: ",st_list
