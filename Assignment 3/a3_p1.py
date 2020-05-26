## Part 1 Final
import sys
import re
import json
import numpy as np
from scipy import stats
from pprint import pprint
from pyspark import SparkContext

sc =SparkContext()
# Load data
rdd = sc.textFile(sys.argv[1])

# Function to get words from Review
def getMap(line1):
    result = []
    line = json.loads(line1)
    try:
        wordList = re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))',line["reviewText"])
        mapp = dict()
        for word in wordList:
            mapp[word.lower()] = mapp.get(word.lower(), 0) + 1
        for x in mapp.items():
            result.append((x[0].lower(),x[1]))
        return result
    except KeyError:
        return []


#Finding top 1000 words
rdd_top = rdd.flatMap(lambda line: getMap(line)).reduceByKey(lambda a,b: a+b).takeOrdered(1000,lambda a: -a[1])

top1000_words = []
for word in rdd_top:
    top1000_words.append(word[0])

broadcastVar = sc.broadcast(top1000_words)

# Finding the relative frequecies of words
def getFrequency(line1):
    result = []
    line = json.loads(line1)
    try:
        wordList = re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))',line["reviewText"])
        mapp = dict()
        for word in wordList:
            mapp[word.lower()] = mapp.get(word.lower(), 0) + 1
        wordSet = set(wordList)
        for word in broadcastVar.value:
        	if word in set(mapp):
        		if(line["verified"]):
        			result.append((word,(float(mapp.get(word))/len(wordList),line["overall"],float(1))))
        		else:
        			result.append((word,(float(mapp.get(word))/len(wordList),line["overall"],float(0))))
        	else:
        		if(line["verified"]):
        			result.append((word,(float(0),line["overall"],float(1))))
        		else:
        			result.append((word,(float(0),line["overall"],float(0))))
        return result
    except KeyError:
    	for word in broadcastVar.value:
    		result.append((word,(float(0),float(0),float(0))))
    	return result

rdd2 = rdd.flatMap(lambda line: getFrequency(line)).groupByKey().mapValues(list)


#Linear Regression Function

def linear_regression(values):
    x = []
    y = []
    for value in values:
        x.append(value[0])
        y.append(value[1])
    X = np.reshape(x, (len(x), 1))
    Y = np.reshape(y, (len(y), 1))
    mean_x = np.mean(X)
    sdev_x = np.std(X)
    mean_y = np.mean(Y)
    sdev_y = np.std(Y)
    X = (X - mean_x) / sdev_x
    Y = (Y - mean_y) / sdev_y
    X = np.hstack((X, np.ones((len(x), 1))))   
    X_inv = np.linalg.pinv(np.dot(np.transpose(X), X))
    w = np.dot(X_inv, np.transpose(X))
    weights = np.dot(w, Y)
    df = len(x) - 2
    rss = np.sum(np.power((Y - np.dot(X, weights)), 2))
    s_squared = rss / df
    se = np.sum(np.power((X[:, 0]), 2))
    tt = (weights[0, 0] / np.sqrt(s_squared / se))
    p_value = stats.t.sf(np.abs(tt), df) * 2
    # Correcting for bonferroni p-value for 1000 words
    return weights[0, 0], p_value * 1000

rdd2 = rdd.flatMap(lambda line: getFrequency(line)).groupByKey().mapValues(list)

newRdd = rdd2.mapValues(linear_regression)
pos_corr = newRdd.takeOrdered(20, lambda x: -x[1][0])
neg_corr = newRdd.takeOrdered(20, lambda x: x[1][0])

def multivariate_regression(values):
    x = []
    y = []
    for value in values:
    	x.append([value[0], value[2]])
    	y.append(value[1])
    
    X = np.reshape(x, (len(x), 2))
    Y = np.reshape(y, (len(y), 1))
    mean_x = np.mean(X, axis=0)
    sdev_x = np.std(X, axis=0)
    mean_y = np.mean(Y)
    sdev_y = np.std(Y)
    if len(x) > 1:
        X = (X - mean_x) / sdev_x
        Y = (Y - mean_y) / sdev_y
    
    X = np.hstack((X, np.ones((len(x), 1))))
    X_inv = np.linalg.inv(np.dot(np.transpose(X), X))
    weights = np.dot(np.dot(X_inv, np.transpose(X)), Y)
    df = len(x) - 3
    rss = np.sum(np.power((Y - np.dot(X, weights)), 2))
    s_squared = rss / df
    se = np.sum(np.power((X[:, 0]), 2))
    tt = (weights[0, 0] / np.sqrt(s_squared / se))
    pval = stats.t.sf(np.abs(tt), df) * 2
    # Correcting for bonferroni p-value for 1000 words
    return (weights[0, 0], pval * 1000)

newRdd1 = rdd2.mapValues(multivariate_regression)
pos_corr1 = newRdd1.takeOrdered(20, lambda x: -x[1][0])
neg_corr1 = newRdd1.takeOrdered(20, lambda x: x[1][0])

print("Top 20 word positively correlated with rating")
pprint(pos_corr)
print("Top 20 word negatively correlated with rating")
pprint(neg_corr)
print("Top 20 word positively correlated with rating, controlling for verified")
pprint(pos_corr1)
print("Top 20 word negatively correlated with rating, controlling for verified")
pprint(neg_corr1)

