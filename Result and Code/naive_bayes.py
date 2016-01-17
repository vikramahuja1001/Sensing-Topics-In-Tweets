import csv
import math
import random
import os
import re
import nltk


def get_data():
	files = ['Fifa.txt','USelections.txt','SuperTuesday.txt']
	files1 = ['Fifa_1.txt','USelections_1.txt','SuperTuesday_1.txt']
	index1 = [0,1,2]
	a = []
	for i in range(len(files)):
		fo = open(files[i] ,'r')
		data = fo.readlines()
		for j in data:
			data1 = [j,index1[i]]
			a.append(data1)

	print len(a)
	return a


def tokenise(data):
	tok = []
	for i in data:
		chars = re.findall("i.e.|Dr.|Mr.|Mrs.|Inc.|Cir.|St.|Jr.|U.S.|N.A.S.A|text-align|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|www\.\w+\.\w+\.?\w+|\w+\.?\w+@\w+\.?\w+\.?\w+|\s[a-zA-Z]\.\s|[\w]+|\"\.|[\+\-()\"=;:*\.,\?!@#$%^&`~'|\\/<>]|\d+%|[0-9]+(?:st|nd|rd|th)",i[0])
		a = [chars,i[1]]
		tok.append(a)
	return tok

def prop_noun(token):
	m=re.match(r'[A-Z]\w+\.?',token)
	return m

def get_unigram(tok_data):
	unigram={}
	for i in tok_data:
		for j in i:
			if j.lower() not in unigram:
				unigram[j.lower()] = 1
			else:
				unigram[j.lower()] +=1
	return unigram



def naive_bayes(test,test_index,train,train_index):
#print dataset
	class_count = [0.0] * 3
	for i in range(len(train)):
		class_count[train_index[i]] +=1.0
	total = 0
	for i in class_count:
		total += i

	print class_count
	#print count_class1
	print total

#--training---
	print "Training"

	prob = [{},{},{}]

	#print prob
	for i in range(len(train)):
		index =  train_index[i]
		for j in train[i]:
			if j not in prob[index]:
				if prop_noun(j) == 0:
					prob[index][j] = 1
				else:
					prob[index][j] = 1
			else:
				if prop_noun(j) == 0:
					s = prob[index][j]
					s +=1
					prob[index][j] = s
				else:
					s = prob[index][j]
					s +=1
					prob[index][j] = s
	
	for i in prob:
		for k, v in i.items():
			if v == 1 or v == 2:
				del i[k]
				#print i[j]



	#print prob
	print 
	for i in range(len(prob)):
		for j in prob[i]:
			#print prob[i][j],class_count[i],i
			prob[i][j] = prob[i][j]/class_count[i]
			#print prob[i][j]
			prob[i][j] = math.log(prob[i][j])
	#print prob
	print total
	for i in range(len(class_count)):
		class_count[i] = class_count[i]/total
	print class_count
	for i in prob:
		print len(i)

#------------------testing--------------------
	print "Testing"
	print len(test)
	count = 0
	for i in range(len(test)):
		p = [0.0] * 3
		for j in test[i]:
			for k in range(3):
				if j in prob[k]:
					p[k] += prob[k][j]
		for j in range(len(p)):
			p[j] += class_count[j] 
		#print p

		a =  p.index(min(p))
		if a == test_index[i]:
			count +=1.0
	print count
	print count/len(test)


#----main
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
print "Getting data"
data1 = []
index = []
data = get_data()
for i in data:
	data1.append(i[0])
	index.append(i[1])
print data1
print index

#print data
print "Tokenizing"
tok_data = []
for i in data1:
	a = nltk.word_tokenize(i)
	tokens = [w for w in a if not w in stopset]
	tok_data.append(tokens)
#tok_data = tokenise(data)
#print tok_data
print len(tok_data)
data = tok_data

print "converting to lower"
data = []
for i in tok_data:
	a = []
	for j in i:
		a.append(j.lower())
	data.append(a)
print len(data)

print "get unigram"
unigram = get_unigram(data)


print "sorting"
res =[]
res = sorted(unigram, key=unigram.get, reverse=True)
res2=[]
for i in range(len(res)):
    res2.append(unigram[res[i]])

#print tok_data[0]
#removing stop words

print len(res)
"""
count = 0
for i in range(len(res)):
	if res2[i] == 1 or res2[i] == 2:
		count +=1
res = res[:-count]
res2 = res2[:-count]
print len(res)
"""

#80% train 20%testing
#Total = 6123
test_len = 60*6123/100
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

for i in range(len(data)):
	if i in te:
		test.append(data[i])
		test_index.append(index[i])
	else:
		train.append(data[i])
		train_index.append(index[i])

print len(test),len (test_index)
print len(train), len(train_index)

naive_bayes(test,test_index,train,train_index)

