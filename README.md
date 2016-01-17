Identifying topics in Tweets is a major problem as it poses the following challenges: the length of a Tweet is very small , noisy data and uneven lingo of social media(Use of words like “U”, “em” etc).
We explored the following methods to detect topics in Tweets and reported our comparisons:
1. LDA (Latent Dirichlet Allocation)
2. Naive Bayes Classifier
3. K-Nearest Neighbor
4. K-Means
5. Boosted N-Grams.
We noticed that KNN performed the best in a 2-fold cross validation setting. We also added boosting to the dataset such that "Hashtags" and proper nouns were doubled in each tweet to expand the tweet similar to query expansion. This boosting drastically improved results through out all classifiers
