"""
  Copyright (c) 2019 - Present – Thomson Licensing, SAS
  All rights reserved.
 
  This source code is licensed under the Clear BSD license found in the
  LICENSE file in the root directory of this source tree.
 """

"""Deploys recommendation algorithms and outputs the recommendations list"""

import pandas as pd
import numpy as np
import os, sys
from signal import *
import pickle
import copy
from surprise import SVD, SVDpp, NMF, KNNBasic
import surprise
from collections import defaultdict
import tempfile
from pprint import pprint
import random
import argparse
import time
import multiprocessing
from functools import partial
import subprocess
import shutil
import glob
import sklearn.metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
import evaluation

sys.path.append(
    os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)) + '/parsers')
import splitter

import builtins as __builtin__

#def print(*args, **kwargs):
#  return __builtin__.print(*args, flush=True, **kwargs)


def surpriseTesting():
  """scikit-surprise library testing"""
  # Load the movielens-100k dataset (download it if needed),
  # and split it into 3 folds for cross-validation.
  data = surprise.Dataset.load_builtin('ml-100k')

#  reader = surprise.Reader(line_format='user item rating', sep=',')
#  data = Dataset.load_from_file('temp.csv', reader=reader)

  trainSet = data.build_full_trainset()
  data.split(n_folds=3)
  for rating in data.build_full_trainset().all_ratings():
    print(rating)

  print(trainSet.n_items)
  algo = SVD()
#  algo = KNNBasic()
  algo.fit(trainSet)
  # Evaluate performances of our algorithm on the dataset.
  perf = surprise.evaluate(algo, data, measures=['RMSE', 'MAE'])

  surprise.print_perf(perf)
  uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
  iid = str(242)  # raw item id (as in the ratings file). They are **strings**!

  # get a prediction for specific users and items.
  pred = algo.predict(uid, iid, r_ui=-1, verbose=True)
  print(pred.est)

def surprisetopN(algo, trainSet, raw_uid, N):
  """Derive the topN recommendations for user uid

  algo: scikit-surprise trained algorithm
  trainSet (surprise.dataset.Trainset)
  raw_uid (int or float): raw uid

  e.g. surprisetopN(algo, trainSet, 196, 3)

  Returns:
    list: (raw_iid, prediction) for the N recommended item_ids

  """
  inner_uid = trainSet.to_inner_uid(raw_uid)
  recom = []
  profile = set(map(lambda x: x[0], trainSet.ur[inner_uid]))
  for iid in trainSet.all_items():
    if iid not in profile: # item is unseen
      raw_iid = trainSet.to_raw_iid(iid)

      pred = algo.predict(raw_uid, raw_iid, r_ui=-1, verbose=False)
      recom.append((raw_iid, pred.est))

  recom = sorted(recom, key=lambda x: x[1], reverse=True)
  return recom[:N]

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    e.g. top_n = get_top_n(predictions, n=10)

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def replay_recom(uid, data, algo=SVD(), thres=sys.maxsize, N=5, cold_start=False):
  """Get topN recommendations for every click of uid
    i.e. remove uid profile from trainSet
       add first item click to trainSet -> train -> get topN
       add second item -> train -> get topN
       ...

  Parameters:

    uid (str): specifies the user (raw uid)
    algo (surprise.prediction_algorithms.algo_base.AlgoBase): training algorithm
    data (pandas.DataFrame): full dataset with uid, iid, rating, timestamps
      !! uid, iid are ints (in surprise-Dataset they are strings)
    thres (int): number of clicks to replay from the user profile
    N (int): top-N recommendations
    cold_start (boolean): whether to add each click to the dataset
      if True => no click is added

  e.g. prof, recom = replay_recom(259, df, 2)

  Returns:
    list: user profile as [(iid1, r1, ts1), ...] sorted by timestamp
    list: topN recom for every click
      [[(iid11, pred11), ... (iid1N, pred1N)],
       [(iid21, pred21), ... (iid2N, pred2N)],
       ...]
  """
  print("N: ", N)
  print("thres: ", thres)
  print("cold_start: ", cold_start)
  print(algo)
  pprint(vars(algo))

  # save algorithm object for initialization after each click
  algoCP = copy.deepcopy(algo)

  # build user profile sorted by timestamp
  profile = [] # list with (item_id, rating, timestamp) enties
  df = data[data['user_id'] == uid].sort_values('timestamp')
  for tup in df.itertuples():
    profile.append((tup.item_id, tup.rating, tup.timestamp))

  profile = profile[:thres] # prune profile with the thres

  # clear user profile from user-item matrix
  temp = data[data['user_id'] != uid]

  # replay the clicks
  recom = [] # topN recommendations for every click
  tf = tempfile.NamedTemporaryFile() # create temp file for saving and loading trainingSet
  print("Saving/Loading to: " + tf.name)
  for i in range(0, len(profile)):
    iid = profile[i][0]
    rating = profile[i][1]
    ts = profile[i][2]
    print("Clicked: ", iid)

    # append fresh click to the dataset
    new = pd.DataFrame([[uid, iid, rating, ts]], columns=('user_id', 'item_id', 'rating', 'timestamp'))
    temp = temp.append(new)

    # save dataframe to file
    temp[['user_id', 'item_id', 'rating']].to_csv(tf.name, index=False,header=False)

    # read it to surprise-dataset
    reader = Reader(line_format='user item rating', sep=',')
    dataSet = Dataset.load_from_file(tf.name, reader=reader)

    # build training and test set
    trainSet = dataSet.build_full_trainset()
    testSet = trainSet.build_anti_testset()

    #algo = KNNBasic()
    #algo = SVD()
    algo = copy.deepcopy(algoCP) # reset algorithm

    # Evaluate performances of the algorithm on the dataset.
    if i == 0: # evaluate only the first time
      dataSet.split(n_folds=3)
      perf = surprise.evaluate(algo, dataSet, measures=['RMSE', 'MAE'])

    # train
    algo.fit(trainSet)

    # predict ratings for all pairs (u, i) that are NOT in the training set.
    predictions = algo.test(testSet)
#    print(predictions[:1000])

    # get topN recommendations
    top = get_top_n(predictions, n=N)[str(uid)] # string uid
    recom.append(top)
    print("TopN: ", top)

    if cold_start:
      temp = data[data['user_id'] != uid] # cold-start scenario for every click

  return (profile, recom)

def surpriseParallelTopNComputation(trainSet, testSet, algoCP, N, update_freq,
                                    appendTest, total, jobID):
  """
  Computes the topN recommendation for a range depending on the jobID
  Args:
    trainSet, testSet (pandas.DataFrame): dataset split with uid, iid, rating, timestamps
    total (int): total number of processes
    jobID (int): process id \in [0, total)
      Defines which testSet rows (i.e., clicks) to derive the topN for
    See surprise_recom() for the rest Args

  Returns:
    (list): [(click_idx1, topn_list1), ...]
      click_idxi: click index on the testSet
      topn_listi: [(reci1_iid, predi1), ... (reci1_iidiN, prediN)]
  """
  step = int(len(testSet) / total)
  mod = len(testSet) % total
  assert jobID * step <= len(testSet)
  # determine start click
  if jobID < mod: # this worker gets an extra click
    start = jobID * step + jobID * 1 # prev jobs' ranges + prev jobs' extra clicks
    end = start + step + 1
  else:
    start = jobID * step + mod # prev jobs' ranges + all extra clicks
    end = start + step

  print("Starting jobID:%d\nProcessing clicks: %s" % (jobID, [start, end-1]))

  topRecoms = []
  hits = 0
  for i in range(start, end):
    # ensure same execution regardless of the # processes
    random.seed(42)
    np.random.seed(42)

    click = testSet.iloc[i]

    if i == start:
      temp_fin = start - (start % update_freq)
      # initialize train and test set
      temp = trainSet.copy()
#      print("jobID%d temp_fin: %d" % (jobID, temp_fin))
      #if appendTest: temp = temp.append(testSet.iloc[0:temp_fin+1], sort=False)
      if appendTest: temp = temp.append(testSet.iloc[0:temp_fin+1])

    # put clicks from testSet to trainSet
#    print("jobID%d Appending: %d, %d " % (jobID, i-(update_freq-1),i+1))
    #if appendTest: temp = temp.append(testSet.iloc[i-(update_freq-1):i+1], sort=False)
    if appendTest: temp = temp.append(testSet.iloc[i-(update_freq-1):i+1])

    # parse to surprise-dataset
    if appendTest or i == start:
      reader = surprise.Reader(rating_scale=(1, 5))
      surTrainDataset = surprise.Dataset.load_from_df(temp[['user_id', 'item_id', 'rating']], reader)
      surTrainSet = surTrainDataset.build_full_trainset()

    # train
    if i % update_freq == 0 or i == start:
      print("Training...")
      algo = copy.deepcopy(algoCP) # reset algorithm
      algo.fit(surTrainSet) # bottleneck

    # get topN recommendations
    top = surprisetopN(algo, surTrainSet, click['user_id'], N)
    print("JobID:%d\tRemaining clicks: %d" % (jobID, end - 1 - i))
    print("JobID:%d\nTopN for click %d: [%s, %s, %s, %s]\n%s" % (
      jobID, i, click['user_id'], click['item_id'], click['rating'], \
      click['timestamp'], top))
    topRecoms.append((i, top))

  assert len(topRecoms) == end - start
  print("Finished for range: (%d, %d)" % (start, end-1))
  return topRecoms

def baseline_recom(trainSet, testSet, N_list=[5]):
  """Implements MovieAvg and Top-Popular from "Performance of recommender
  algorithms on top-N recommendation tasks" paper
  See surprise_recom() for args, returns"""
  print("N_list: %s" % N_list)
  s_time = time.time()
  print("Start time: %s" % s_time)

  print("Training size: %s" % len(trainSet))
  print("Test size: %s" % len(testSet))

  # create item-set sorted by average rating
  itemSet = trainSet.groupby('item_id')['rating'].agg(['count', 'mean'])
#  itemSet = itemSet.sort_values('mean', ascending=False).reset_index() # MovieAvg
  itemSet = itemSet.sort_values('count', ascending=False).reset_index() # Top Popular

  # create seen items
  seenSet = trainSet.groupby('user_id')['item_id'].agg(lambda x:
                                                      set(x)).reset_index()

  # create return recommendation list
  res = []
  hits = 0
  for index, row in testSet.iterrows():
    topn = []
    i = 0
    while len(topn) < max(N_list) and i < len(itemSet):
      seenItems = seenSet[seenSet['user_id'] ==
                          row['user_id']]['item_id'].values[0]
      if itemSet.iloc[i]['item_id'] not in seenItems:
        topn.append((itemSet.iloc[i]['item_id'], max(N_list) - len(topn)))
      i += 1

    res.append((row['user_id'], row['item_id'], row['timestamp'], topn))

  for N in N_list:
    evaluation.offlineEvaluation(res, N)

  print("Finish time: %s" % time.time())
  print("Total time: %s" % (time.time() - s_time))

  return res




def surprise_recom(trainSet, testSet, algo=SVD(), drop_ratio=0, N_list=[5], \
    update_freq=1, appendTest=True, num=multiprocessing.cpu_count(), evalTrain=True):
  """Get topN recommendations for every click in the test set

  Args:
    trainSet (pandas.DataFrame): training dataset with uid, iid, rating, timestamps
    testSet (pandas.DataFrame): test dataset with uid, iid, rating, timestamps
    algo (surprise.prediction_algorithms.algo_base.AlgoBase): training algorithm
    drop_ratio (float): how many rows to drop from trainSet at random
      if \in (0, 1) => drop drop_ratio * len(trainSet) rows
      if > 1 => drop drop_ratio rows
    N_list (list): top-N = max(N_list)
      the rest of the N-values are only used for the evaluation
    update_freq (int): after how many clicks to update the model
    appendTest (boolean): incrementally append test ratings to training set
      while getting topN and re-training; simulates online learning
    num (int): number of processes for parallelizing topN computation
    evalTrain (boolean): evaluate on training set with 5-fold cross validation

  Returns:
    (list): topN recom for every click
      [(click_uid1, click_iid1, click_ts1, topn_list1), ...]
        topn_listi: [(reci1_iid, score11), ... (reci1_iidiN, scoreiN)]
      If no timestamp (ts) is available => click_ts = click order
  """
  print("N_list: %s" % N_list)
  print("drop_ratio: %s" % drop_ratio)
  print("update_freq: %s" % update_freq)
  print("appendTest: %s" % appendTest)
  print("Eval Train: %s" % evalTrain)
  print(algo)
  pprint(vars(algo))
  s_time = time.time()
  print("Start time: %s" % s_time)

  # save algorithm object for initialization after each click
  algoCP = copy.deepcopy(algo)

  # drop random rows from trainSet
  if drop_ratio < 1:
    drop_indices = np.random.choice(
        trainSet.index, drop_ratio * len(trainSet), replace=False)
  else:
    drop_indices = np.random.choice(
        trainSet.index, drop_ratio, replace=False)

  trainSet = trainSet.drop(drop_indices)
  print("Training size: %s" % len(trainSet))
  print("Test size: %s" % len(testSet))
  
  # parse dataset to suprise Dataset format
  reader = surprise.Reader(rating_scale=(1, 5))
  surTrainDataset = surprise.Dataset.load_from_df(trainSet[['user_id', 'item_id', 'rating']], reader)
  surTrainSet = surTrainDataset.build_full_trainset()
  
  if evalTrain:
    print("Evaluating recommendation algorithm with 5-fold KV on train \\union test")
    surTrainDataset.split(n_folds=5)
    perf = surprise.evaluate(algo, surTrainDataset, measures=['RMSE', 'MAE'])

  tmpdir = tempfile.mkdtemp()
  print("Created temp dir: " + tmpdir)
  print("If SIGTERM or SIGINT temp dir will be wiped out;" + \
      "otherwise must remove manually")
  # remove tmp file in case of interrupt
  handler = partial(cleanDir, tmpdir)
  for sig in (SIGABRT, SIGILL, SIGINT, SIGSEGV, SIGTERM):
    signal(sig, handler)
  # save dataframe to file
  trainSet[['user_id', 'item_id', 'rating']].to_csv(
      tmpdir + '/trainSet.csv', index=False, header=False)
  testSet[['user_id', 'item_id', 'rating']].to_csv(
      tmpdir + '/testSet.csv', index=False, header=False)

  print("Evaluating recommendation algorithm on specified split")
  reader = surprise.Reader(line_format='user item rating', sep=',')
  trainTestData = surprise.Dataset.load_from_folds(
      [(tmpdir + '/trainSet.csv', tmpdir + '/testSet.csv')], reader=reader)
  perf = surprise.evaluate(algo, trainTestData, measures=['RMSE', 'MAE'])

  # Parallel topN computation
  part = partial(surpriseParallelTopNComputation, trainSet, testSet, \
      algoCP, max(N_list), update_freq, appendTest, num)
  jobIDs = range(0, num)
  print("Spawning %d processes for topN computation..." % num)
  pool = multiprocessing.Pool(num)
  outputs = pool.map_async(part, jobIDs).get(timeout=9999999) # enable killing with SIGTERM
  pool.close()
  pool.join()

  # flatten output
  res = []
  for output in outputs:
    for item in output:
      res.append(item)

  # sort output based on click ordering
  res = sorted(res, key=lambda x: x[0])

  # create return recommendation list
  # TODO If no timestamp (ts) is available => click_ts = click order
  res = list(map(lambda t: (testSet.iloc[t[0]]['user_id'], \
    testSet.iloc[t[0]]['item_id'], testSet.iloc[t[0]]['timestamp'], t[1]),
                 res))

  for N in N_list:
    if appendTest:
      evaluation.onlineEvaluation(res, N)
    else:
      evaluation.offlineEvaluation(res, N)

  clicked = list(map(lambda t: t[1], res))
  print("Distinct clicks: ", len(set(clicked)))

  avg_score = 0
  count = 0
  for uid, iid, ts, topn in res:
    for r in topn:
      avg_score += r[1]
      count += 1
  print("Average score: ", avg_score / float(count))

  assert len(clicked) == len(testSet['item_id'])

  print("Finish time: %s" % time.time())
  print("Total time: %s" % (time.time() - s_time))

  return res


def cleanDir(tmpdir, signal, frame):
  print("DEATH. Cleaning tmpdir...")
  shutil.rmtree(tmpdir)
  sys.exit(0)

def main(args):

  parser = argparse.ArgumentParser(description= \
      'Deploys recommendation algorithms and outputs the recommendations list',\
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--pickleLoadPath", type=str, action='store', \
      help= 'If set=> load topN recoms from pickle file')
  parser.add_argument("--pickleSavePath", type=str, action='store',
      help= 'If set => Output .pickle file.')

  parser.add_argument("--proc", type=int, default=multiprocessing.cpu_count(), \
      action='store', \
      help= 'Number of processes to spawn for topN computation\n' +
      'default is number of processors.')
  parser.add_argument("--update_freq", type=int, default=1, action='store', \
      help= 'Number of clicks after which the model is updated')
  parser.add_argument("--topN_list", type=int, nargs="+", required=True, \
      help= 'e.g., --topN_list 5 10 50\n' \
      + 'topN=max(topN_list); the rest of the values are used for evaluation.')
  parser.add_argument("--drop_ratio", type=int, default=0, action='store', \
      help= 'Number of random events to remove from the training set;\n' + \
      'default is 0.')
  parser.add_argument("--evalTrain", dest='evalTrain', action='store_true', \
      help='If set => evaluate on training set using k-fold validation.\n' \
          + 'Else => evaluate only on test set')

  parser.add_argument("--dataset", type=str, action='store', \
      help= 'Full path to the dataset.\n' + \
      'Must give --testSize and --validSize for the split')
  parser.add_argument("--testSize", type=int, default=0, action='store',
      help= 'TestSet size; default is 0 => no test set')
  parser.add_argument("--validSize", type=int, default=2000, action='store', \
      help= 'Validation Set size; default is 2000.')
  parser.add_argument("--trainSet", type=str, action='store', \
      help= 'Full path to the trainingSet.csv\n' + \
      'If given the (potential) training set split from --dataset will be overwritten')
  parser.add_argument("--validSet", type=str, action='store', \
      help= 'Full path to the validationSet.csv\n' + \
      'If given the (potential) validation set split from --dataset will be overwritten')
  parser.add_argument("--testSet", type=str, action='store', \
      help= 'Full path to the testSet.csv\n' + \
      'If given the (potential) test set split from --dataset will be overwritten')

  parser.add_argument("--surprise_algo", type=str, action='store', \
      help= 'Choose algorithm from surprise lib. Available options:\n' + \
      '--surprise_algo SVD\n' + \
      '--surprise_algo SVDpp\n' + \
      '--surprise_algo PMF\n' + \
      '--surprise_algo NMF\n' + \
      '--surprise_algo KNNWithMeans\n')

  args = parser.parse_args(args)

  random.seed(42) # reproducability
  np.random.seed(42)

  if args.pickleLoadPath is None:

    """DATA"""
    train, valid, test = splitter.splitData(
          fullDataPath=args.dataset, validSize=args.validSize, testSize=args.testSize, \
          trainSetPath=args.trainSet, validSetPath=args.validSet, testSetPath=args.testSet)

    """RECOMMENDATIONS"""
    if args.surprise_algo == 'SVD':
      algo = surprise.SVD()
    elif args.surprise_algo == 'KNNWithMeans':
#     sim_options = {'name': 'pearson_baseline', 'shrinkage': 2500, \
#        'user_based': False, }
      sim_options = {'name': 'cosine', 'user_based': False}
      algo = surprise.KNNWithMeans(k=40, sim_options=sim_options)
    elif args.surprise_algo == 'PMF':
      algo = surprise.SVD(n_factors=5, reg_all=0.12, lr_all=0.005, n_epochs=400)
    elif args.surprise_algo == 'NMF':
      algo = surprise.NMF(n_factors=5, n_epochs=400)
    elif args.surprise_algo == 'SVDpp':
      algo = surprise.SVDpp()

    testList = [] # output recommendations for the last element
    if len(test) > 0:
      testList.append(test)
    if len(valid) > 0:
      testList.append(valid)

    for test in testList:
        recs = surprise_recom(train, test, algo, drop_ratio=args.drop_ratio, \
            update_freq=args.update_freq, N_list=args.topN_list, num=args.proc, \
            evalTrain=args.evalTrain)

    if not args.pickleSavePath is None:
      with open(args.pickleSavePath, 'wb') as handle:
        pickle.dump(recs, handle)

  else:
    with open(args.pickleLoadPath, 'rb') as handle:
      recs = pickle.load(handle)

if __name__ == "__main__":
  main(sys.argv[1:])
