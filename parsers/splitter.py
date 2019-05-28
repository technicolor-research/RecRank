"""
  Copyright (c) 2019 - Present – Thomson Licensing, SAS
  All rights reserved.
 
  This source code is licensed under the Clear BSD license found in the
  LICENSE file in the root directory of this source tree.
 """

"""Derives the training, validation and test set for the recommendation algorithm"""

import pandas as pd
import pickle
import numpy as np
import sys
from sklearn.model_selection import train_test_split

def parseDataset(path):
  """Parses dataset from .csv file to a pandas.DataFrame"""

  header = ['user_id', 'item_id', 'rating', 'timestamp']
  df = pd.read_csv(path, sep='\t', names=header)
  n_users = df.user_id.unique().shape[0]
  n_items = df.item_id.unique().shape[0]
  print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
  sparsity=round(1.0-len(df)/float(n_users*n_items), 6)
  print('Sparsity level for '+path+' : ' +  str(sparsity*100) + '%')
  sys.stdout.flush()

  return df

def randomSplit(path):
  """Follows "Performance of Recommender Algorithms on Top-N Recommendation Tasks"
  Args: path to .csv file with ratings
  Returns:
    pandas.DataFrame: trainSet
    pandas.DataFrame: testSet
  """
  data = parseDataset(path)
  trainSet, probeSet = train_test_split(data, test_size=0.014, random_state=42)
  testSet = probeSet[probeSet['rating'] == 5].reset_index()

  # filter unknown test users
  trainUsers = set(trainSet['user_id'])
  testSet = testSet[testSet['user_id'].isin(trainUsers)].reset_index()

  return trainSet, testSet


def filteredDataSplit(fullDataPath, testSetSize):
  """Derives train and test set:
    Finds the optimal splits s.t. testSet doesn't contain any users in the trainingSet
      and testSet size is the closest one to the testSetSize
  Returns:
    pandas.DataFrame: trainSet
    pandas.DataFrame: testSet
  """

  data = parseDataset(fullDataPath)
  data = data.sort_values('timestamp')
  data['user_id'] = data['user_id'].astype(str) # make uids strings
  data['item_id'] = data['item_id'].astype(str) # make iids strings

  split_ratio = testSetSize # how many examples to put in the testSet
  optimal_split = (0, testSetSize)  # (split_ratio, |len(testSet) - testSetSize|)
  loopsWithNoUpdates = 0
  while True:
    dataSplit = np.split(data, [len(data) - int(split_ratio)], axis=0)
    trainSet = dataSplit[0]
    testSet = dataSplit[1]

    trainUsers = set(trainSet['user_id'])
    dropIdx = [] # indices (i.e., ratings) with user_id \in trainUsers
    for idx, row in testSet.iterrows():
      if row['user_id'] in trainUsers:
        dropIdx.append(idx)
    testSet = testSet.drop(dropIdx)

    print("Split: %d\tTestSize: %d" % (split_ratio, len(testSet)))
    # update optimal split ratio
    if optimal_split[1] > abs(len(testSet)-testSetSize):
      print("Update")
      optimal_split = (split_ratio, abs(len(testSet)-testSetSize))
      loopsWithNoUpdates = 0
    else:
      loopsWithNoUpdates += 1

    if len(testSet) < testSetSize: # move the split to the left
      split_ratio += testSetSize - len(testSet)
    elif len(testSet) > testSetSize: # move the split to the right
      split_ratio -= len(testSet) - testSetSize
    else:
      break
    if loopsWithNoUpdates == 20: # flunctuates for 10 iterations
      break

    sys.stdout.flush()

  print("Optimal split-ratio: %d\tDiff: %d" % optimal_split)
  sys.stdout.flush()
  dataSplit = np.split(data, [len(data) - int(optimal_split[0])], axis=0)
  trainSet = dataSplit[0]
  testSet = dataSplit[1].iloc[:testSetSize]

  return trainSet, testSet

def leaveXOutSplit(fullDataPath, x=1):
  """Offline protocol (leave-one-out)
  https://iscs.nus.edu.sg/~kanmy/papers/sigir16.pdf <=> leaveXOutSplit(path, 1)
  Returns:
    pandas.DataFrame: trainSet
    pandas.DataFrame: testSet
  """
  data = parseDataset(fullDataPath)
  data = data.sort_values('timestamp')
  data['user_id'] = data['user_id'].astype(str) # make uids strings
  data['item_id'] = data['item_id'].astype(str) # make iids strings

  # pick the test clicks
  testSet = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
  dropIdx = [] # idx for test examples
  user_set = list(data['user_id'].unique()) # users with no ratings on the test
  user_set = user_set * x
  for idx, row in data[::-1].iterrows():
    if row['user_id'] in user_set:
      testSet = testSet.append(row)
      dropIdx.append(idx)
      user_set.remove(row['user_id'])
    if len(user_set) == 0:
      break

  trainSet = data.drop(dropIdx)

  return trainSet, testSet

def splitData(trainSize=None, validSize=None, testSize=None, \
    fullDataPath=None, \
    trainSetPath=None, validSetPath=None, testSetPath=None, \
    filterTest=False, shuffle=False):
  """Derives training and test set

  Args:
    trainSize (float): size of the training set
      if \in (0,1): e.g. ratio=0.8 => trainSize = 80% of dataset
      if > 1: e.g. ratio=6000 => trainSize = 6000 examples
    validSize (float): size of the validation set
    testSize (float): size of the test set
      ! If 2 of the 3 sizes given => size3 = len(data) - size1 - size2

    fullDataPath (string): if set read full dataset from file and split
    trainSetPath (string): if set read the training set from file
    validSetPath (string): if set read the validation set from file
    testSetPath (string): if set read the test set from file
    filterTest (boolean): keeps only new_user's ratings on the testSet
    shuffle (boolean): shuffle the dataset before splitting

  Returns:
    pandas.DataFrame: trainSet
    pandas.DataFrame: validSet
    pandas.DataFrame: testSet
  """
  if fullDataPath:
    data = parseDataset(fullDataPath)

    if shuffle:
      data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
      data = data.sort_values('timestamp')
    data['user_id'] = data['user_id'].astype(str) # make uids strings
    data['item_id'] = data['item_id'].astype(str) # make iids strings

    # parse sizes
    if not trainSize is None and trainSize < 1:
      trainSize = int(round(trainSize * len(data)))
    if not validSize is None and validSize < 1:
      validSize = int(round(validSize * len(data)))
    if not testSize is None and testSize < 1:
      testSize = int(round(testSize * len(data)))

    # determine 3rd size if the other 2 are given
    if trainSize is None and not validSize is None and not testSize is None:
      trainSize = len(data) - validSize - testSize
    elif not trainSize is None and validSize is None and not testSize is None:
      validSize = len(data) - trainSize - testSize
    elif not trainSize is None and not validSize is None and testSize is None:
      testSize = len(data) - trainSize - validSize
    else:
      raise ValueError("Must give at least 2 out of 3 Sizes (train,valid,test)")

    # split
    trainSet, validSet, testSet, _ = np.split(data, \
        [trainSize, trainSize+validSize, len(data)], axis=0)

  if trainSetPath:
    print("Reading training set from file")
    trainSet = parseDataset(trainSetPath)
    trainSet = trainSet.sort_values('timestamp')
    trainSet['user_id'] = trainSet['user_id'].astype(str) # make uids strings
    trainSet['item_id'] = trainSet['item_id'].astype(str) # make iids strings

  if validSetPath:
    print("Reading validation set from file")
    validSet = parseDataset(validSetPath)
    validSet = validSet.sort_values('timestamp')
    validSet['user_id'] = validSet['user_id'].astype(str) # make uids strings
    validSet['item_id'] = validSet['item_id'].astype(str) # make iids strings

  if testSetPath:
    print("Reading test set from file")
    testSet = parseDataset(testSetPath)
    testSet = testSet.sort_values('timestamp')
    testSet['user_id'] = testSet['user_id'].astype(str) # make uids strings
    testSet['item_id'] = testSet['item_id'].astype(str) # make iids strings

  if filterTest:
    trainUsers = set(trainSet['user_id'])
    validUsers = set(validSet['user_id'])
    dropIdx = [] # indices (i.e., ratings) with user_id \in trainUsers
    for idx, row in testSet.iterrows():
      if row['user_id'] in trainUsers or row['user_id'] in validUsers:
        dropIdx.append(idx)
    testSet = testSet.drop(dropIdx)

  if fullDataPath or (trainSetPath and validSetPath and testSetPath):
    print("Train, valid, test split: %s %s %s" % (
      len(trainSet), len(validSet), len(testSet)))

  return trainSet, validSet, testSet

def getUnknownQuery(known_train, known_test, unknown_train):
  """Derives the set to query the black-box algorithm (unknown_query)
  based on the known_dataset and the itemSet of the unknown_dataset

  Args:
    known_train (pandas.DataFrame): user_id, item_id, rating, timestamp
    known_test (pandas.DataFrame)
    unknown_train (pandas.DataFrame)

  Returns:
    pandas.DataFrame: unknown_test
  """
  # get ranked by popularity (rating sum) itemset
  # item_id -> ratingSum (sorted by ratingSum)
  knownItemSums = known_train[['item_id', 'rating']] \
      .groupby(['item_id'], sort=False).sum().sort_values('rating', ascending=False)

  unknownItemSums = unknown_train[['item_id', 'rating']] \
      .groupby(['item_id'], sort=False).sum().sort_values('rating', ascending=False)

  unknown_query = pd.DataFrame()
  for idx, row in known_test.iterrows():
    if row['item_id'] in knownItemSums.index: # not new item
      # get item rank on known_dataset
      known_rank = knownItemSums.index.get_loc(row['item_id'])
      # get unknown rank (map knownItems to unknownItems)
      sizeRatio = len(knownItemSums) / float(len(unknownItemSums))
      unknown_rank = int(round(known_rank / sizeRatio))

      row['item_id'] = unknownItemSums.index[unknown_rank]
    unknown_query = unknown_query.append(row)

  return unknown_query

def getKnownQuerySet(known_train, unknownTopN):
  """Derives the set to query the known algorithms based on the querySet of the
  black-box. The unknownTopN contains queries for the most popular items.
  Mapping unknown -> known:
    ith-popular unknown-> ith-popular known
    uid -> uid
    rating -> max rating
    timestamp -> timestamp

  Args:
    known_train (pandas.DataFrame): user_id, item_id, rating, timestamp
    unknownTopN (list): see return of imdbQuery.py#getTopN

  Returns:
    pandas.DataFrame: knownQuerySet
  """
  # get ranked by popularity (rating sum) itemset
  # item_id -> ratingSum (sorted by ratingSum)
  knownItemSums = known_train[['item_id', 'rating']] \
      .groupby(['item_id'], sort=False).sum().sort_values('rating', ascending=False)
  known_query = pd.DataFrame(
      columns=['user_id', 'item_id', 'rating', 'timestamp'])
  known_query['user_id'] = known_query['user_id'].astype(str) # make uids strings
  known_query['item_id'] = known_query['item_id'].astype(str) # make iids strings

  rating = 5 # TODO get max rating
  for idx in range(0, len(unknownTopN)):
    if (idx >= knownItemSums.shape[0]):
      break
    uid, iid, ts, topn_list = unknownTopN[idx]
    # get idx-th most popular item on known_dataset
    known_query = known_query.append(pd.DataFrame([[
      uid, knownItemSums.iloc[idx].name, rating, ts]],
      columns=list(known_query.columns)))

  return known_query

#def randomizeTest(testSet):
#  """Randomizes the test examples but preserves the rating frequency
#  for every user. Useful for breaking the grouped-by-user ratings of movielens
#  Args:
#    testSet (pandas.DataFrame): user_id, item_id, rating, timestamp
#  Returns:
#    pandas.DataFrame: randomized testSet
#  """
#  userRatings = {} # uid -> [r1, r2, ...]
#  for idx, row in testSet.iterrows():
#    if row['user_id'] not in userRatings:
#      userRatings[row['user_id']] = [row['rating']

def main(args):
  """Manual split"""

  data = parseDataset('ML100K.csv')
  data = data.sort_values('timestamp')
  #print(data)
  data['user_id'] = data['user_id'].astype(str) # make uids strings
  data['item_id'] = data['item_id'].astype(str) # make iids strings

	# * test7-8 *
#  evalSet = data[-4001:] # evaluation sets
#  trainSet = data[:-4001] # training sets
#
#  unknown_test = evalSet[-2000:]
#  known_val = evalSet[-4001:-2000]
#  known_query = unknown_test.sample(frac=0.5, random_state=42).reset_index(drop=True)
#
#  # shuffle trainSet
#  trainSet = trainSet.sample(frac=1, random_state=42).reset_index(drop=True)
#  print(trainSet)
#  known_train = trainSet[:10000]
#  unknown_train = trainSet[10000:]
#
#  print(known_train)
#  print(unknown_train)
#  known_train = known_train.sort_values('timestamp')
#  unknown_train = unknown_train.sort_values('timestamp')
	# * test7-8 *

	# * test9 *
  known_train = data[:4000]
  known_val = data[4000:5000]
  unknown_train = data[5000:-2000]
  unknown_test = data[-2000:]
  known_query = unknown_test.sample(frac=0.5, random_state=42).reset_index(drop=True)
	# * test9 *

  print(known_train)
  print(unknown_train)
  print(known_val)
  print(unknown_test)
  print(known_query)

  known_train.to_csv('known_train.csv', header=False, index=False)
  known_val.to_csv('known_val.csv', header=False, index=False)
  known_query.to_csv('known_query.csv', header=False, index=False)
  unknown_train.to_csv('unknown_train.csv', header=False, index=False)
  unknown_test.to_csv('unknown_test.csv', header=False, index=False)

if __name__ == "__main__":
  main(sys.argv[1:])
