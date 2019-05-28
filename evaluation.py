"""
  Copyright (c) 2019 - Present – Thomson Licensing, SAS
  All rights reserved.
 
  This source code is licensed under the Clear BSD license found in the
  LICENSE file in the root directory of this source tree.
 """


"""Evaluates the output (i.e., topN recs list) of a recommender"""

import numpy as np
import sklearn.metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y


def my_ndcg_score(y_true, y_score, k=5):
  """Overrides sklearn.metrics.ndcg_score"""
  y_score, y_true = check_X_y(y_score, y_true)

  # Make sure we use all the labels (max between the length and the higher
  # number in the array)
  lb = LabelBinarizer()
  lb.fit(np.arange(max(np.max(y_true) + 1, len(y_true))))
  binarized_y_true = lb.transform(y_true)

  # !! GEORGIOS DAMASKINOS FIX due to problematic shape caused by:
    # LabelBinarizer: Binary targets transform to a column vector
  if len(y_true) == 2:
    binarized_y_true = np.hstack((1-binarized_y_true, binarized_y_true))

  if binarized_y_true.shape != y_score.shape:
      raise ValueError("y_true and y_score have different value ranges")

  scores = []

  # Iterate over each y_value_true and compute the DCG score
  for y_value_true, y_value_score in zip(binarized_y_true, y_score):
      actual = sklearn.metrics.dcg_score(y_value_true, y_value_score, k)
      best = sklearn.metrics.dcg_score(y_value_true, y_value_true, k)
      scores.append(actual / best)

  return np.mean(scores)

def user_ndcg(user_clicks, rank):
  """Calculates ndcg for the given user
  Args: evaluation#recom filter the given user
  Returns:
    float: nDCG@rank score
  """
  # create mapping: iid -> integer
  mapping = {}
  i = 0
  user = user_clicks[0][0]
  for uid, iid, ts, topn in user_clicks:
    assert uid == user # clicks - recommendations only for a single user
    assert iid not in mapping # each user clicks each item at most once
    mapping[iid] = i
    i += 1

#  print("Total distinct items clicked | recommended to this user: %s" % i)

  y_true = [] # ground truth (see sklearn ndcg_score)
  y_score = [] # predicted scores (-//-)
  for uid, iid, ts, topn in user_clicks:
    y_true.append(mapping[iid])
    sample_score = [0] * i
    for iid2, score in topn:
      if iid2 in mapping:
        sample_score[mapping[iid2]] = score
    y_score.append(sample_score)

  return my_ndcg_score(y_true, y_score, k=rank)

def onlineEvaluation(recom, rank=5):
  """Calculates evaluation metrics
  Assumes that no item is clicked more than once by the same user
  Args:
    recom (list): see surprise_recom return arguement
    rank (int): ranking threshold
  Returns:
    float: precision according to I-SIM paper
    float: recall    -//-
    float: ndcg according to sklearn
  """
  print("Evaluation @%s" % rank)
  hits = {}
  clicked = {}
  recommended = {}
  for i in range(0, len(recom)):
    uid = recom[i][0]
    iid = recom[i][1]
    ts = recom[i][2]
    topn = recom[i][3]

    if uid not in clicked: # if user is unseen
      hits[uid] = set()
      recommended[uid] = set()
      clicked[uid] = set()

    clicked[uid].add(iid) # append click to clicked
    for r, score in topn[:rank]:
      recommended[uid].add(r) # append recommendation to recommended
      # find hits by checking all clicks
      for j in range(0, len(recom)):
        if uid == recom[j][0]: # if click belongs to the same user
          if j <= i: # sanity check for not recommending seen items
            if r == recom[j][1]:
              print("PROBLEM")
              print(r)
              print(recom[i])
              print(recom[j])
            assert r != recom[j][1]
          else:
            # if user clicks this item
            if r == recom[j][1]:
              hits[uid].add(r)

  precision = {}
  recall = {}
  #ndcg = {}
  totalRecom = 0
  totalClicks = 0
  totalHits = 0
  #nanCount = 0
  for uid in recommended.keys():
    totalRecom += len(recommended[uid])
    totalClicks += len(clicked[uid])
    totalHits += len(hits[uid])

    precision[uid] = len(hits[uid]) / float(len(recommended[uid]))
    recall[uid] = len(hits[uid]) / float(len(clicked[uid]))

   # user_recom = list(filter(lambda x: x[0] == uid, recom))
   # if len(user_recom) > 1: # len=1 => ndcg = nan
   #   ndcg[uid] = user_ndcg(user_recom, rank)
   # else:
   #   nanCount += 1

  avg_precision = np.mean(list(precision.values()))
  avg_recall = np.mean(list(recall.values()))

  f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
  if np.isnan(f1):
    f1 = 0

  #print("Clicked@%s: %s" % (rank, totalClicks))
  #print("Recommended@%s: %s" % (rank, totalRecom))
  #print("Hits@%s: %s" % (rank, totalHits))

  print("Precision@%s: %s\n%s" % (rank, avg_precision, precision))
  print("Recall@%s: %s\n%s" % (rank, avg_recall,recall))
  print("F1@%s: %s" % (rank, f1))
  #print("nDCG@%s: %s\n%s\nNanCount: %s" % (rank, avg_ndcg, ndcg, nanCount))

  return avg_precision, avg_recall #, avg_ndcg


def offlineEvaluation(recom, rank=5):
  """Calculates evaluation metrics according to: https://www.researchgate.net/profile/Paolo_Cremonesi/publication/221141030_Performance_of_recommender_algorithms_on_top-N_recommendation_tasks/links/55ef4ac808ae0af8ee1b1bd0.pdf
  Args:
    recom (list): see surprise_recom return arguement
    rank (int): ranking threshold
  Returns:
    float: precision
    float: recall    -//-
  """
  print("Evaluation @%s" % rank)
  hits = {}
  clicked = {}
  recommended = {}
  for i in range(0, len(recom)):
    uid = recom[i][0]
    iid = recom[i][1]
    ts = recom[i][2]
    topn = recom[i][3]

    if uid not in clicked: # if user is unseen
      hits[uid] = set()
      recommended[uid] = set()
      clicked[uid] = set()

    clicked[uid].add(iid) # append click to clicked
    for r, score in topn[:rank]:
      recommended[uid].add(r) # append recommendation to recommended
      if r  == iid:
        hits[uid].add(r)

  precision = {}
  recall = {}
  totalRecom = 0
  totalClicks = 0
  totalHits = 0
  for uid in recommended.keys():
    totalRecom += len(recommended[uid])
    totalClicks += len(clicked[uid])
    totalHits += len(hits[uid])

    precision[uid] = len(hits[uid]) / float(len(recommended[uid]))
    recall[uid] = len(hits[uid]) / float(len(clicked[uid]))

  #avg_precision = np.mean(list(precision.values()))
  #avg_recall = np.mean(list(recall.values()))
  avg_recall = totalHits / totalClicks
  avg_precision = avg_recall / rank

  if avg_precision + avg_recall == 0:
    f1 = 0
  else:
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

  print("Clicked@%s: %s" % (rank, totalClicks))
  print("Recommended@%s: %s" % (rank, totalRecom))
  print("Hits@%s: %s" % (rank, totalHits))
  print("Precision@%s: %s" % (rank, avg_precision))
  print("Recall@%s: %s" % (rank, avg_recall))
  print("F1@%s: %s" % (rank, f1))

  return avg_precision, avg_recall

