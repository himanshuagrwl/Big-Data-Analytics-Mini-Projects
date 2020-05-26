## Part 2 
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
listofItem = eval(sys.argv[2])


#Filtering : per user per item
def fun(line):
    line = json.loads(line)
    return ((line["reviewerID"],line["asin"]),(line["overall"]))

rdd_filter1 = rdd.map(lambda line: fun(line)).reduceByKey(lambda a,b: b)

# Filter to items associated with at least 25 distinct users (91 items)

rdd_filter2 = rdd_filter1.map(lambda a: (a[0][1],(a[0][0],a[1]))).groupByKey().filter(lambda a: len(a[1])>=25).mapValues(list)

# Filter to users associated with at least 5 distinct items (142 users)

#Format (user,list[item,rating])
user_rdd = rdd_filter2.flatMap(lambda a: [(record[0],(a[0],record[1])) for record in a[1]]).groupByKey().filter(lambda a: len(a[1])>=5).mapValues(list)

#Re-Format (item,list[user,rating])
item_rdd = user_rdd.flatMap(lambda a: [(record[0],(a[0],record[1])) for record in a[1]]).groupByKey().mapValues(list)

# Function to compute cosine similarity
def find_similar(curr_item_list):
    SimValue = 0.0
    s = set([record[0] for record in curr_item_list]).intersection(user_broadcast.value.keys())
    if len(s) >= 2:
	    d_curr = dict((k,v) for (k,v) in curr_item_list)
	    mean = np.mean(list(d_curr.values()))
	    d_curr.update((k,v-mean) for (k,v) in d_curr.items())
	    numerator = 0.0
	    for val in s:
	        numerator += user_broadcast.value[val] * d_curr[val]
	    denominator = np.sqrt(np.sum(np.square(list(user_broadcast.value.values())))) * np.sqrt(np.sum(np.square(list(d_curr.values()))))
	    SimValue = float(numerator) / denominator
    return SimValue

# Function to find the predicted rating
def calculate(prod, user_list, d2):
    d1 = dict((k,v) for (k,v) in user_list)
    if prod in d1.keys():
        return d1[prod]
    numerator = 0.0
    denominator = 0.0
    for key in d2.keys():
        numerator += d1[key] * d2[key]
        denominator += d2[key]
    try:
        return float(numerator) / denominator
    except Exception as e:
        return 0.0
    return 0.0


for item in listofItem:
	user_list_item = item_rdd.filter(lambda a: a[0] == item).flatMap(lambda a: a[1]).collectAsMap()
	mean = np.mean(list(user_list_item.values()))
	user_list_item.update((k,v-mean) for (k,v) in user_list_item.items())
	user_broadcast = sc.broadcast(user_list_item)
	similarity_map = item_rdd.map(lambda a: (a[0], find_similar(a[1]))).filter(lambda a: a[1] > 0).collectAsMap()
	print(" User list for ",item,": ")
	# Find target row values
	pprint(user_rdd.filter(lambda a: len(set([record[0] for record in a[1]]).intersection(set(similarity_map))) >= 2).map(lambda a: (a[0], a[1] ,dict((rec, similarity_map[rec]) for rec in set([record[0] for record in a[1]]).intersection(similarity_map.keys())))).map(lambda a:(a[0],calculate(item,a[1],a[2]))).filter(lambda a: a[1]>0).collect())
