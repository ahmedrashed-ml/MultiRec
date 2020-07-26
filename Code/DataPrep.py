import numpy as np
import pandas as pd
import time
from collections import deque

import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
import random
import pickle
import math
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("../Data/ebay/auction.csv", sep=',', engine='python')
df.bidder=df.bidder.astype(str)

auctionidEncoder= preprocessing.LabelEncoder()
BidderEncoder= preprocessing.LabelEncoder()

df.auctionid= auctionidEncoder.fit_transform(df.auctionid)
df.bidder= BidderEncoder.fit_transform(df.bidder)

df.bidder=df.bidder.astype(int)
df.auctionid=df.auctionid.astype(int)



dfItemFeatures=df[['auctionid','openbid','item','auction_type']].drop_duplicates(subset='auctionid')
dfItemFeatures.auctionid=dfItemFeatures.auctionid.astype(int)
dfDealerFeatures= df[['bidder','bidderrate']].drop_duplicates(subset='bidder')
dfDealerFeatures.bidder=dfDealerFeatures.bidder.astype(int)

df=df.sort_values(by=['auctionid','bid'],ascending=False)


PurchasedItems=[]
PurchaseMatrixList=[]
BiddingMatrixList=[]

for index, row in df.iterrows():
  auctionid=int(row['auctionid'])
  bid=float(row['bid'])
  price=float(row['price'])
  bidder=int(row['bidder'])
  if(bid==price) and (auctionid not in PurchasedItems):
    PurchasedItems.append(auctionid)
    PurchaseMatrixList.append([bidder,auctionid,price])
  else:
    BiddingMatrixList.append([bidder,auctionid,bid])

PurchaseMatrix=np.asarray(PurchaseMatrixList)
BiddingMatrix=np.asarray(BiddingMatrixList)
print(PurchaseMatrix.shape)
print(BiddingMatrix.shape)
print(dfDealerFeatures.shape)
print(dfItemFeatures.shape)

np.savetxt("../Data/ebay/Ratings.csv", PurchaseMatrix, delimiter=";")
np.savetxt("../Data/ebay/Biddings.csv", BiddingMatrix, delimiter=";")
dfItemFeatures.to_csv("../Data/ebay/ItemFeatures.csv", sep=';', encoding='utf-8', index=False)
dfDealerFeatures.to_csv("../Data/ebay/DealerFeatures.csv", sep=';', encoding='utf-8', index=False)



