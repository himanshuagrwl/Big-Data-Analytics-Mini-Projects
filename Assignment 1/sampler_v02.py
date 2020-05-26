##########################################################################
## Simulator.py  v 0.1
##
## Implements two versions of a multi-level sampler:
##
## 1) Traditional 3 step process
## 2) Streaming process using hashing
##
##
## Original Code written by H. Andrew Schwartz
## for SBU's Big Data Analytics Course 
## Spring 2020
##
## Student Name: Himanshu Agrawal


##Data Science Imports:
import numpy as np
import mmh3
import random

##IO, Process Imports:
import sys
from pprint import pprint
from datetime import datetime


##########################################################################
##########################################################################
# Task 1.A Typical non-streaming multi-level sampler

def typicalSampler(filename, percent=.01, sample_col=0):
    # Implements the standard non-streaming sampling method
    # Step 1: read file to pull out unique user_ids from file
    # Step 2: subset to random  1% of user_ids
    # Step 3: read file again to pull out records from the 1% user_id and compute mean withdrawn
    # <<COMPLETE>>
    mean, standard_deviation = 0.0, 0.0
    userid = set()
    for line in filename:
        userid.add(line.split(',')[sample_col])
    useridlist = list(userid)
    random.shuffle(useridlist)
    sampled_userid = [useridlist[i] for i in range(0, int(len(useridlist) * percent))]
    filename.seek(0)
    count = 0
    for line in filename:
        U_id, amt = line.split(',')[sample_col], line.split(',')[3]
        if U_id in sampled_userid:
            count += 1
            new_mean = mean + ((float(amt) - mean) / count)
            new_standard_deviation = standard_deviation + (float(amt) - new_mean) * (float(amt) - mean)
            mean = new_mean
            standard_deviation = new_standard_deviation

    return mean, np.sqrt(standard_deviation / count)


##########################################################################
##########################################################################
# Task 1.B Streaming multi-level sampler

def streamSampler(stream, percent=.01, sample_col=0):
    # Implements the standard streaming sampling method:
    #   stream -- iosteam object (i.e. an open file for reading)
    #   percent -- percent of sample to keep
    #   sample_col -- column number to sample over
    #
    # Rules:
    #   1) No saving rows, or user_ids outside the scope of the while loop.
    #   2) No other loops besides the while listed.

    mean, standard_deviation = 0.0, 0.0
    noOfBuckets = 1 / percent
    chooseBucket = random.randint(0, noOfBuckets - 1)
    count = 0
    ##<<COMPLETE>>
    for line in stream:
        U_id, amt = line.split(',')[sample_col], line.split(',')[3]
        if mmh3.hash(U_id) % noOfBuckets == chooseBucket:
            count += 1
            new_mean = mean + ((float(amt)-mean)/count)
            new_standard_deviation = standard_deviation + (float(amt)-new_mean) * (float(amt)-mean)
            mean = new_mean
            standard_deviation = new_standard_deviation

    return mean, np.sqrt(standard_deviation/count)


##########################################################################
##########################################################################
# Task 1.C Timing

files = ['transactions_small.csv', 'transactions_medium.csv', 'transactions_large.csv']
percents = [0.02, .005]

if __name__ == "__main__":

    ##<<COMPLETE: EDIT AND ADD TO IT>>
    for perc in percents:
        print("\nPercentage: %.4f\n==================" % perc)
        for f in files:
            print("\nFile: ", f)
            fstream = open(f, "r")
            start = datetime.now()
            print("  Typical Sampler: ", typicalSampler(fstream, perc, 2))
            end = datetime.now()
            print("  Time = ", (end - start).total_seconds() * 1000)
            fstream.close()
            fstream = open(f, "r")
            start = datetime.now()
            print("  Stream Sampler:  ", streamSampler(fstream, perc, 2))
            end = datetime.now()
            print("  Time = ", (end - start).total_seconds() * 1000)
