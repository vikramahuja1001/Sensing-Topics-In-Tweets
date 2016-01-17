import re
import math
from random import randint
rnd = []
while len(rnd) != 0:
	a= randint(0,1900)
	if a not in rnd:
		rnd.append(a)
train_ans = []
data=open('Fifa_1.txt','r').readlines()
tok = []
chars_fifa = []
for i in range(len(data)):
	#print i
	chars = re.split(" ", data[i])
	chars = chars[:-1]
	for j in range(len(chars)):
		chars[j] = chars[j].lower()
		if i not in rnd:
			tok.append(chars[j])
	chars_fifa.append(chars)
#print chars_fifa

test = []
a = []
for i in range(len(chars_fifa)):
	if i in rnd:
		test.append(chars_fifa[i])
	else:
		a.append(chars_fifa[i])
chars_fifa = []

for i in range(len(a)):
	chars_fifa.append(a[i])

#Unigrams_fifa
unigram_fifa={}
for i in tok:
    if i not in unigram_fifa:
        unigram_fifa[i] = 1
    else:
        unigram_fifa[i] +=1

res_fifa=[]
res_fifa = sorted(unigram_fifa, key=unigram_fifa.get, reverse=True)
res2_fifa=[]
res3_fifa=[]
for i in range(len(res_fifa)):
    res2_fifa.append(unigram_fifa[res_fifa[i]])
    res3_fifa.append(res2_fifa[i])

#res contains sorted unigrams and res2 contain their count
print "FIFA Top 15:"
for i in range(15):
	print res_fifa[i],res3_fifa[i]



#US Election
data=open('USelections_1.txt','r').readlines()
tok = []
chars_use=[]
for i in range(len(data)):
	#print i
	chars = re.split(" ", data[i])
	chars = chars[:-1]
	for j in range(len(chars)):
		chars[j] = chars[j].lower()
		if i not in rnd:
			tok.append(chars[j])
	chars_use.append(chars)
	#print chars

a = []
for i in range(len(chars_use)):
	if i in rnd:
		test.append(chars_use[i])
	else:
		a.append(chars_use[i])

chars_use=[]
for i in range(len(a)):
	chars_use.append(a[i])

#Unigrams_fifa_1
unigram_use={}
for i in tok:
    if i not in unigram_use:
        unigram_use[i] = 1
    else:
        unigram_use[i] +=1

res_use=[]
res_use = sorted(unigram_use, key=unigram_use.get, reverse=True)
res2_use=[]
res3_use=[]
for i in range(len(res_use)):
    res2_use.append(unigram_use[res_use[i]])
    res3_use.append(res2_use[i]*1.0)

print 
#res contains sorted unigrams and res2 contain their count
print "USE Top 15:"
for i in range(15):
	print res_use[i],res3_use[i]



#Super Tuesday
data=open('SuperTuesday_1.txt','r').readlines()
tok = []
chars_st=[]
for i in range(len(data)):
	#print i
	chars = re.split(" ", data[i])
	chars = chars[:-1]
	for j in range(len(chars)):
		chars[j] = chars[j].lower()
		if i not in rnd:
			tok.append(chars[j])
	chars_st.append(chars)
	#print chars

a = []
for i in range(len(chars_st)):
	if i in rnd:
		test.append(chars_st[i])
	else:
		a.append(chars_st[i])

chars_st = []

for i in range(len(a)):
	chars_st.append(a[i])

#Unigrams_fifa_1
unigram_st={}
for i in tok:
    if i not in unigram_st:
        unigram_st[i] = 1
    else:
        unigram_st[i] +=1

res_st=[]
res_st = sorted(unigram_st, key=unigram_st.get, reverse=True)
res2_st=[]
res3_st=[]
for i in range(len(res_st)):
    res2_st.append(unigram_st[res_st[i]])
    res3_st.append(res2_st[i]*1.0)

print 
#res contains sorted unigrams and res2 contain their count
print "ST Top 15:"
for i in range(15):
	print res_st[i],res3_st[i]


#Smoothing left
print len(res_fifa)
print len(res_use)
print len(res_st)

print
print len(chars_fifa)
print len(chars_use)
print len(chars_st)
fifa_len = len(chars_fifa)
use_len = len(chars_use)
st_len = len(chars_st)
total_len = fifa_len + use_len + st_len

#schema -> fifa->0,use->1,
print "Calculating FIFA TF"
tf = {}
for i in range(len(res_fifa)):
	if res_fifa[i] not in tf:
		num = []
		posn = []
		final = []
		for j in range(len(chars_fifa)):
			count = 0
			#print chars_fifa[j]
			for k in chars_fifa[j]:
				#print k
				if k == res_fifa[i]:
					count += 1
			if count != 0:
				num.append(count)
				posn.append(j)
				final = [num,posn]
		tf[res_fifa[i]] = final

print "Calculating USE TF"
for i in range(len(res_use)):
	if res_use[i] not in tf:
		num = []
		posn = []
		final= []
		for j in range(len(chars_use)):
			count = 0
			#print chars_fifa[j]
			for k in chars_use[j]:
				#print k
				if k == res_use[i]:
					count += 1
			if count != 0:
				num.append(count)
				posn.append(j+fifa_len)
				final = [num,posn]
		tf[res_use[i]] = final
	else:
		final = tf[res_use[i]]
		num =  tf[res_use[i]][0]
		posn =  tf[res_use[i]][1]
		for j in range(len(chars_use)):
			count = 0
			#print chars_fifa[j]
			for k in chars_use[j]:
				#print k
				if k == res_use[i]:
					count += 1
			if count != 0:
				num.append(count)
				posn.append(j+fifa_len)
				final = [num,posn]
		tf[res_use[i]] = final


print "Calculating ST TF"
for i in range(len(res_st)):
	if res_st[i] not in tf:
		num = []
		posn = []
		final= []
		for j in range(len(chars_st)):
			count = 0
			#print chars_fifa[j]
			for k in chars_st[j]:
				#print k
				if k == res_st[i]:
					count += 1
			if count != 0:
				num.append(count)
				posn.append(j+fifa_len+use_len)
				final = [num,posn]
		tf[res_st[i]] = final
	else:
		final = tf[res_st[i]]
		num =  tf[res_st[i]][0]
		posn =  tf[res_st[i]][1]
		for j in range(len(chars_st)):
			count = 0
			#print chars_fifa[j]
			for k in chars_st[j]:
				#print k
				if k == res_st[i]:
					count += 1
			if count != 0:
				num.append(count)
				posn.append(j+fifa_len+use_len)
				final = [num,posn]
		tf[res_st[i]] = final



tf.pop("")
print len(tf)
#print tf
idf = {}
#print tf['obama']
print "Calculating IDF"
for i,j in tf.iteritems():	
	idf[i] = 1 + (math.log(total_len/(len(j[0])*1.0))/math.log(math.e))


tfidf = {}
for i in tf:
	tfidf[i] = tf[i]
print "Calculating TFIDF"
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='k-means++', max_iter=5000, n_init=1)

km.fit_predict(tfidf,y=None)
print km.labels_
print len(km.labels_)

for i ,j in tfidf.iteritems():
	a = []
	b = j[0]
	c = j[1]
	for k in range(len(j[0])):
		b[k] = (b[k] * idf[i])
	a.append(b)
	a.append(c)
	tfidf[i] = a

#Testing starts
#Making taining points
print "Testing"

print len(test)
print test[0]
for i in range(len(test)):
	j = 0
	while j!=len(test[i]):
		if test[i][j] not in tfidf:
			#print test[i][j]
			test[i].remove(test[i][j])
		else:
			j +=1
			#problem above

print test[0] 
#test[0] =  ['obama', 'won' ,'the', 'fucking','election']

#a =  ['obama', 'won' ,'the', 'fucking','election','chelsea']
"""
for i in a:
	print i
	print tfidf[i]
"""


def cos_similarity(vec1,vec2):
	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])
	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	denominator = math.sqrt(sum1) * math.sqrt(sum2)
	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator


def cosine_similarity(query,doc):
	dot = 0.0
	for i in range(len(query)):
		dot += query[i]*doc[i]
	#dot = sum(p*q for p,q in zip(query, doc))
	q=0.0
	d=0.0
	for i in range(len(query)):
		q +=(query[i]*query[i])
		d +=(doc[i]*doc[i])
	q = math.sqrt(q)
	d = math.sqrt(d)
	if q*d == 0.0:
		return 0
	return dot/(q*d)



tfidf_final = {}
s = 0 
for k in test:
	for i in range(len(k)):
		#print k[i]
		if k[i] not in tfidf_final:
			num = []
			for j in range(total_len):
				num.append(0.0)
			tfidf_final[k[i]
			] = num
			#print len(tfidf[k[i]][0])
			#print len(num)

			for j in range(len(tfidf[k[i]][0])):
				num[tfidf[k[i]][1][j]] = tfidf[k[i]][0][j]
			tfidf_final[k[i]] = num
#print tfidf['election']
#print tfidf_final['election']


	train_score = []
	for i in range(len(k)):
		train_score.append(tfidf_final[k[i]])

	tf_test={}
	for i in range(len(k)):
		if k[i] not in tf_test:
			tf_test[k[i]] = 1.0
		else:
			tf_test[k[i]] += 1
	for i in range(len(k)):
		tf_test[k[i]] /= len(k)

	print tf_test
	idf_test={}
	for i in range(len(k)):
		if k[i] not in idf_test:
			idf_test[k[i]] = idf[k[i]]
	print idf_test

	tfidf_test = {}
	for i in range(len(k)):
		if k[i] not in tfidf_test:
			tfidf_test[k[i]] = tf_test[k[i]] * idf_test[k[i]]
	print tfidf_test

	test_score = []
	for i in range(len(k)):
		test_score.append(tfidf_test[k[i]])

	#train_score = []
	result = []
	max_ = 0.0
	index = 0
	for i in range(total_len):
		train_score = []
		for j in range(len(k)):
			train_score.append(tfidf_final[k[j]][i])
		#print train_score
		b = cosine_similarity(train_score,test_score)
		if b > max_:
			max_ = b
			index = i
			result.append(index)
	print len(result)
	for i in range(len(result)):
		print result[i]
		#print b

	print max_,index
	#break
	#for i in range(len(train_score)):
	#a = cos_similarity(train_score[0],test_score)
	#print train_score[0]
	#print test_score
	#b = cosine_similarity(train_score[0],test_score)

	#print b
	break


	unique_pts = []
	for i in range(total_len):
		a = ''
		for j in range(len(train_score)):
			a += str(train_score[j][i])
		if a not in unique_pts:
			unique_pts.append(a)
	s += len(unique_pts)
	print len(unique_pts)
s = s*1.0
print s/len(test)


#print test[0]
"""
for k in test:
	print k
	tfidf_final = {}
	for i in range(len(k)):
		print k[i]
		num = []
		if k[i] not in tfidf_final:
			for j in range(total_len):
				num.append(0.0)
			for j in range(len(tfidf[k[i]][0])):
				num[tfidf[k[i]][1][j]] = tfidf[k[i]][0][j]
			tfidf_final[k[i]] = num
	print tfidf_final['obama']
	print tfidf_final['the']
	train_score = []
	for i in range(len(k)):
		train_score.append(tfidf_final[k[i]])

	unique_pts = []
	for i in range(total_len):
		a = ''
		for j in range(len(train_score)):
			a += str(train_score[j][i])
		if a not in unique_pts:
			unique_pts.append(k)
	for i in range(len(test)):
		if test[i] == k:
			break
	print len(unique_pts),i
	break
"""

"""
import matplotlib.pyplot as plt1
plt1.axis([0, 100, 0, 100])		
plt1.scatter(a,b,color = 'blue')
plt1.show()
"""
#print a
#print b
#plt1.plot([1,2,3,4], [1,4,9,16], 'ro')
#plt1.axis([-0.001, 0.02, -0.001, 0.02])
#plt1.show()
#Training Done

"""
#obama won
print tfidf['obama']
print tfidf['fuck']

test_tf = [0.5,0.5]
test_idf = [1 + (math.log(2)/math.log(math.e)),1.0 ]
test_tfidf = []
for i in range(len(test_tf)):
	test_tfidf.append(  test_tf[i] * test_idf[i])
print test_tfidf
a = [0.0,  0.0009309017052426691]
b = [0.046531288679407555, 0.0024622716441620334]

def cosine_similarity(query,doc):
	dot = sum(p*q for p,q in zip(query, doc))
	q = math.sqrt(query[0] * query[0] + query[1]*query[1])
	d = math.sqrt(doc[0]*doc[0] + doc[1]*doc[1])
	return dot/(q*d)

print cosine_similarity(test_tfidf,a)
print cosine_similarity(test_tfidf,b)

"""


