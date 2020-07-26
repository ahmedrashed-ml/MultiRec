import os 
import numpy as np
import random
import tensorflow as tf
import sys
import pandas as pd
import time
from collections import deque
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
from sklearn.metrics.pairwise import cosine_similarity
import datetime
SEED=int(sys.argv[1])
UseSalePrice=int(sys.argv[2])
UseBidding=int(sys.argv[3])
UseValidation=int(sys.argv[4])
np.random.seed(SEED)
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)

DIM=60
USER_NUM = 3388
ITEM_NUM = 628
EPOCH_MAX =2000
BATCH_SIZE = 100
NEGSAMPLES=1
UseUserData=1
UseItemData=1
UseGraphData=0
UserReg=0.0
ItemReg=0.0

print(UseUserData,"-",UseItemData,"-",UseGraphData,"-",DIM,"-",UserReg,"-",ItemReg,"-")


def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x= pickle.load(f)
    except:
        x = []
    return x

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_process(filname, sep="\t"):
    col_names = ["user", "item","price"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    return df


def generate_EbayRankData():
    df = read_process("../Data/ebay/Ratings.csv", sep=";")
    dictUsers={}
    ##Filling all PosTrain
    print('Filling all PosTrain')
    for index, row in df.iterrows():
      userid=int(row['user'])

      if userid in dictUsers:
        dictUsers[userid][0].append(row['item'])
      else:
        #print('no')
         # 0->PosTrain,1->NegTrain,2->TestPos 1,3->Test 100,4->ValidationPos,5->Validation 100
        dictUsers[userid]={0:list(),1:list(),2:list(),3:list(),4:list(),5:list()}
        dictUsers[userid][0].append(row['item'])####000
    print(len(dictUsers[0][0]))

    print('Selecting validation Instance ')    
    ## Selecting Test Instance    
    for userid in dictUsers: 
      if(len(dictUsers[userid][0])>2):
        lastitem = dictUsers[userid][0].pop(len(dictUsers[userid][0])-1)
        dictUsers[userid][4].append(lastitem)####4444

    print('Selecting Test Instance ')    
    ## Selecting Test Instance    
    for userid in dictUsers: 
      if(len(dictUsers[userid][0])>1):
        lastitem = dictUsers[userid][0].pop(len(dictUsers[userid][0])-1)
        dictUsers[userid][2].append(lastitem)####222

    print('Filling Neg Instance ')
    ## Filling Neg Instance          
    for userid in dictUsers: 
      for i in range(ITEM_NUM):
        if i not in dictUsers[userid][0] :
          dictUsers[userid][1].append(i)####111

    
    print('Creating TestSet and add last test item ')
    ## Creating TestSet and add last test item      
    for userid in dictUsers:
      if(len(dictUsers[userid][2])==1):
        if(len(dictUsers[userid][1])>99):
          dictUsers[userid][3]=random.sample(dictUsers[userid][1],k=99)
        else:
          print('yes')
          dictUsers[userid][3]=dictUsers[userid][1]
        dictUsers[userid][3].append(dictUsers[userid][2][0])

    print('Creating ValSet and add last val item ')
    ## Creating TestSet and add last test item      
    for userid in dictUsers:
      if(len(dictUsers[userid][4])==1):
        if(len(dictUsers[userid][1])>99):
          dictUsers[userid][5]=random.sample(dictUsers[userid][1],k=99)
        else:
          print('yes')
          dictUsers[userid][5]=dictUsers[userid][1]
        dictUsers[userid][5].append(dictUsers[userid][4][0])
    
    ##All Training Positive Data
    print('All Training Positive Data')
    TrainItems=list()
    TrainUsers=list()
    TrainTargets=list()
    for userid in dictUsers:
      ItemsLength=len(dictUsers[userid][0])
      TrainUsers.extend(np.repeat(userid, ItemsLength))
      TrainItems.extend(dictUsers[userid][0])
      TrainTargets.extend(np.repeat(1.0, ItemsLength))
    
    
    df_train= pd.DataFrame(columns=['user', 'item', 'rate'])
    df_train['user']=TrainUsers
    df_train['item']=TrainItems
    df_train['rate']=TrainTargets


    return dictUsers, df_train  


def inferenceDense(phase,featuresize,user_batch, item_batch,time_batch,idx_user,idx_item,ureg,ireg, user_num, item_num, dim=25, UReg=0.05,IReg=0.1, device="/cpu:0"):
    with tf.device(device):
        user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
        user_batch=tf.cast(user_batch, tf.float64)
        item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")
        item_batch=tf.cast(item_batch, tf.float64)
        
        ul1mf=tf.layers.dense(inputs=user_batch, units=dim+25,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        il1mf=tf.layers.dense(inputs=item_batch, units=dim+25,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        ul2mf=tf.layers.dense(inputs=ul1mf, units=dim,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        il2mf=tf.layers.dense(inputs=il1mf, units=dim,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        InferInputMF=tf.multiply(ul2mf, il2mf)

        infer=tf.reduce_sum(InferInputMF, 1, name="inference")
        
        pl1=tf.layers.dense(inputs=InferInputMF, units=5,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        #pl2=tf.layers.dense(inputs=pl1, units=10,activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        inferPrice=tf.reduce_sum(pl1, 1, name="inferencePrice")
        regularizer = tf.add(UReg*tf.nn.l2_loss(ul1mf),IReg*tf.nn.l2_loss(il1mf), name="regularizer")
    return infer,inferPrice, regularizer


def optimization(infer,inferPrice,mask_batch,prices_batch, regularizer, rate_batch, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost =tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch,logits=infer)# tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        salecost=tf.reduce_mean(tf.abs(tf.boolean_mask(prices_batch,mask_batch)-tf.boolean_mask(inferPrice,mask_batch)))
        TotalLoss=1*cost+UseSalePrice*0.00008*salecost
        train_op =tf.contrib.opt.AdamWOptimizer(0.00001,learning_rate=learning_rate).minimize(TotalLoss, global_step=global_step)
    return salecost,cost, train_op

def optimizationBidding(infer,mask_batch,prices_batch, regularizer, rate_batch, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost =0.5*tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch,logits=infer)
        train_op =tf.contrib.opt.AdamWOptimizer(0.00001,learning_rate=learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op    

def clip(x):
    return np.clip(x, 1.0, 5.0)

def GetTrainSample(DictUsers,BatchSize=256,negsamples=1):
  trainusers=np.asarray([])
  trainitems=np.asarray([])
  traintargets=np.asarray([])
  numusers=int(BatchSize/(negsamples+1))
  #print(numusers)
  for i in range(numusers):
    batchusers=random.choice(list(DictUsers.keys())) #random.randint(0,USER_NUM-1)
    while len(DictUsers[batchusers][0])==0:
      batchusers=random.choice(list(DictUsers.keys()))

    trainusers=np.append(trainusers,np.repeat(batchusers, negsamples+1))
    ##Pos
    trainitems=np.append(trainitems,np.random.choice(DictUsers[batchusers][0], 1))
    traintargets=np.append(traintargets,[1.0])
    ##Neg
    trainitems=np.append(trainitems,np.random.choice(DictUsers[batchusers][1], negsamples))
    traintargets=np.append(traintargets,np.zeros(negsamples))
    
  return trainusers,trainitems,traintargets




def svd(train,ItemData=False,UserData=False,Graph=False,lr=0.00002,ureg=0.05,ireg=0.02):

    UserFeatures=np.identity(USER_NUM)
    ItemFeatures=[]


    if(UserData):
      UsrDat=get_UserData()
      UserFeatures=np.concatenate((UserFeatures,UsrDat), axis=1) 
      UserFeatures =UserFeatures.astype(np.int16)
      print(UserFeatures.shape)
      del UsrDat

    if(ItemData):
      ItemFeatures=ITEMDATA
      print(ItemFeatures.shape)     

    print(UserFeatures.shape)
    print(ItemFeatures.shape)
    


    samples_per_batch = len(train) // int(BATCH_SIZE/(NEGSAMPLES+1))
    

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    y_batch = tf.placeholder(tf.float64, shape=[None,1], name="y")
    m_batch = tf.placeholder(tf.float64, shape=[None,1], name="m")
    d_batch = tf.placeholder(tf.float64, shape=[None,1], name="d")
    dw_batch = tf.placeholder(tf.float64, shape=[None,1], name="dw")
    dy_batch = tf.placeholder(tf.float64, shape=[None,1], name="dy")
    w_batch = tf.placeholder(tf.float64, shape=[None,1], name="w")
    
    
    time_batch=tf.concat([y_batch, m_batch,d_batch,dw_batch,dy_batch,w_batch], 1)
    rate_batch = tf.placeholder(tf.float64, shape=[None])
    prices_batch = tf.placeholder(tf.float64, shape=[None])
    mask_batch = tf.placeholder(tf.bool, shape=[None])
    phase = tf.placeholder(tf.bool, name='phase')
    shapet=UserFeatures.shape[1]+ItemFeatures.shape[1]

    w_user = tf.constant(UserFeatures,name="userids", shape=[UserFeatures.shape[0],UserFeatures.shape[1]],dtype=tf.int16)#copying everything to gpu memory
    del UserFeatures 
    w_item = tf.constant(ItemFeatures,name="itemids", shape=[ItemFeatures.shape[0], ItemFeatures.shape[1]],dtype=tf.float32)#copying everything to gpu memory
    del ItemFeatures 
    
    #user_onehotbatch=tf.one_hot(user_batch, USER_NUM)
    #item_onehotbatch=tf.one_hot(item_batch, ITEM_NUM)

    infer,inferPrice, regularizer = inferenceDense(phase,shapet,user_batch, item_batch,time_batch,w_user,w_item,ureg,ireg, user_num=USER_NUM, item_num=ITEM_NUM,
                                           device=DEVICE)#these are all placeholders as well
    global_step = tf.contrib.framework.get_or_create_global_step()
    salecost,cost, train_op = optimization(infer,inferPrice,mask_batch,prices_batch, regularizer, rate_batch, learning_rate=lr, reg=0.09, device=DEVICE)
    Bidcost, trainBid_op = optimizationBidding(infer,mask_batch,prices_batch, regularizer, rate_batch, learning_rate=lr, reg=0.09, device=DEVICE)
    

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.05


    with tf.Session(config=config) as sess:
        sess.run(init_op)
        print("epoch, train_err,train_saleerror,HitRatio10,HitRatio20,HitRatio50,NDCG10,NDCG20,NDCG50")
        errors = deque(maxlen=samples_per_batch)
        saleerrors = deque(maxlen=samples_per_batch)
        now = datetime.datetime.now()
        textTrain_file = open("./OutputEbay/"+now.strftime('%Y%m%d%H%M%S')+"_"+str(SEED)+".txt", "w",newline='')
        print(samples_per_batch)
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = GetTrainSample(dictUsers,BATCH_SIZE,NEGSAMPLES)
            usersBids, itemsBids, ratesBids = GetBiddingSample()
            prices=get_salePrice(users,items) 
            mask=prices > 0

            _, pred_batch,cst,slcost = sess.run([train_op, infer,cost,salecost], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   prices_batch:prices,
                                                                   mask_batch:mask,
                                                                   phase:True})
            if(UseBidding):
              _, Bidcst = sess.run([trainBid_op,Bidcost], feed_dict={user_batch: usersBids,
                                                          item_batch: itemsBids,
                                                          rate_batch: ratesBids,
                                                          phase:True})
                                                   
            pred_batch = clip(pred_batch)
            errors.append(cst)
            saleerrors.append(slcost)
            if i % samples_per_batch == 0:
                #print('-----')
                train_saleerror=np.mean(saleerrors)
                train_err = np.mean(errors)
                totalhits=0
                ############
                correcthits10=0
                correcthits20=0
                correcthits50=0
                ndcg10=0
                ndcg20=0
                ndcg50=0

                ###########
                if(UseValidation):
                  for userid in dictUsers: 
                    if(len(dictUsers[userid][4])==0):
                      continue
    
                    items=dictUsers[userid][5]
                    TestItem=dictUsers[userid][4][0]
                    users=np.repeat(userid, 100)
                    items=dictUsers[userid][5]

                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,                                                                                             
                                                            phase:False}) 
                    sorteditems=[x for _, x in sorted(zip(pred_batch,items), key=lambda pair: pair[0],reverse=True)]
                    #######
                    topitems10=sorteditems[:10]
                    correcthits10=correcthits10+getHR(topitems10,TestItem)
                    correcthits20=correcthits20+getHR(sorteditems[:20],TestItem)
                    correcthits50=correcthits50+getHR(sorteditems[:50],TestItem)

                    ndcg10=ndcg10+getNDCG(topitems10,TestItem)
                    ndcg20=ndcg20+getNDCG(sorteditems[:20],TestItem)
                    ndcg50=ndcg50+getNDCG(sorteditems[:50],TestItem)
                    totalhits=totalhits+13
                    ############
                  HitRatio = correcthits10/ totalhits
                  HitRatio20 = correcthits20/ totalhits
                  HitRatio50 = correcthits50/ totalhits
                  NDCG = ndcg10/ totalhits
                  NDCG20 = ndcg20/ totalhits
                  NDCG50 = ndcg50/ totalhits
                  print("{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch, train_err,train_saleerror,HitRatio,HitRatio20,HitRatio50,NDCG,NDCG20,NDCG50))                
                else:
                  testableusers=0

                  for userid in dictUsers: 
                    if(len(dictUsers[userid][2])!=0):
                        testableusers+=1


                  for userid in dictUsers: 
                    if(len(dictUsers[userid][2])==0):
                      continue
                    
                    items=dictUsers[userid][3]
                    TestItem=dictUsers[userid][2][0]

                    users=np.repeat(userid, 100)
                    items=dictUsers[userid][3]

                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,                                                                                             
                                                            phase:False}) 
                    sorteditems=[x for _, x in sorted(zip(pred_batch,items), key=lambda pair: pair[0],reverse=True)]
                    #######
                    topitems10=sorteditems[:10]
                    correcthits10=correcthits10+getHR(topitems10,TestItem)
                    correcthits20=correcthits20+getHR(sorteditems[:20],TestItem)
                    correcthits50=correcthits50+getHR(sorteditems[:50],TestItem)

                    ndcg10=ndcg10+getNDCG(topitems10,TestItem)
                    ndcg20=ndcg20+getNDCG(sorteditems[:20],TestItem)
                    ndcg50=ndcg50+getNDCG(sorteditems[:50],TestItem)
                    totalhits=totalhits+1
                    ############
                  HitRatio = correcthits10/ totalhits
                  HitRatio20 = correcthits20/ totalhits
                  HitRatio50 = correcthits50/ totalhits
                  NDCG = ndcg10/ totalhits
                  NDCG20 = ndcg20/ totalhits
                  NDCG50 = ndcg50/ totalhits
                  print("{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch, train_err,train_saleerror,HitRatio,HitRatio20,HitRatio50,NDCG,NDCG20,NDCG50)) 
                  textTrain_file.write("{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch, train_err,train_saleerror,HitRatio,HitRatio20,HitRatio50,NDCG,NDCG20,NDCG50) +'\n')            
                  textTrain_file.flush()
         

def getHR(ranklist, gtItem): 
    if(gtItem in ranklist):
      return 1 
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def get_UserData():
    df = pd.read_csv('../Data/ebay/DealerFeatures.csv', sep=';', engine='python')
    df=df.sort_values(by=['bidder'])
    print(df.shape)
    del df['bidder']
    values =df.values.astype(np.int32)
    return values
  

def get_ItemData():
    df = pd.read_csv('../Data/ebay/ItemFeatures.csv', sep=';', engine='python')
    df=df.sort_values(by=['auctionid'])
    del df['auctionid']
    print(df.shape)
    df=pd.concat([df,df['item'].str.get_dummies(sep=' ').add_prefix('Name_').astype('int8')],axis=1)    
    df=pd.concat([df,df['auction_type'].str.get_dummies(sep=' ').add_prefix('Auction_').astype('int8')],axis=1) 
    del df['item']
    del df['auction_type']

    print(df.shape)

    df=pd.get_dummies(df,dummy_na=True)
    df=df.fillna(df.mean())
    df=df.dropna(axis=1, how='all')

    print(df.shape)
    values=df.values


    return values


def GetBiddingSample(BatchSize=256,negsamples=1):
  global BIDDINGDATA
  trainusers=np.asarray([])
  trainitems=np.asarray([])
  traintargets=np.asarray([])
  numusers=int(BatchSize/(negsamples+1))
  for i in range(numusers):
    batchusers=random.choice(list(BIDDINGDATA.keys())) 
    while len(BIDDINGDATA[batchusers])==0:
      batchusers=random.choice(list(BIDDINGDATA.keys()))

    trainusers=np.append(trainusers,np.repeat(batchusers, negsamples+1))
    ##Pos
    trainitems=np.append(trainitems,np.random.choice(BIDDINGDATA[batchusers], 1))
    traintargets=np.append(traintargets,[1.0])
    ##Neg
    negitem=random.randint(0,ITEM_NUM-1)
    while negitem in BIDDINGDATA[batchusers]:
      negitem=random.randint(0,ITEM_NUM-1)
    trainitems=np.append(trainitems,negitem)
 
    
    
    traintargets=np.append(traintargets,np.zeros(negsamples))
    
  return trainusers,trainitems,traintargets

def get_salePrice(userarr,itemarr):
    global SALEDATA
    prices=SALEDATA[userarr[:], itemarr[:]]
    return np.asarray(prices).reshape(-1)

def read_SalePrices():    
    col_names = ["user", "item","price"]
    df = pd.read_csv("../Data/ebay/Ratings.csv", sep=';', header=None, names=col_names, engine='python')
    print(df.shape)
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    SALEDATA=csr_matrix( (USER_NUM,ITEM_NUM) )  
    for index, row in df.iterrows():
      #print(index)
      userid=int(row['user'])
      itemid=int(row['item'])
      saleprice=row['price']
      #print(userid,'-',itemid,'-',saleprice)
      SALEDATA[userid,itemid]=saleprice
    return SALEDATA

def read_Bidding(dictU):
  col_names = ["user", "item","bid"]
  dfBidding = pd.read_csv("../Data/ebay/Biddings.csv", sep=';', header=None, names=col_names, engine='python') 
  BiddingDict={}
  for index, row in dfBidding.iterrows():
    userid=int(row['user'])
    itemid=int(row['item'])
    if userid in dictU :
      if len(dictU[userid][2])==1:
        if userid in BiddingDict:
          if itemid != dictU[userid][2][0]:
            BiddingDict[userid].append(itemid)

        else:
          BiddingDict[userid]=list()
          if itemid != dictU[userid][2][0]:
            BiddingDict[userid].append(itemid)
      else:
        if userid in BiddingDict:
          BiddingDict[userid].append(itemid)

        else:
          BiddingDict[userid]=list()
          BiddingDict[userid].append(itemid)

  return BiddingDict

LEARNRATE=0.00003
DIM=60
DEVICE = "/cpu:0"

dictUsers,df_train=generate_EbayRankData()
SALEDATA=read_SalePrices()
BIDDINGDATA=read_Bidding(dictUsers)
ITEMDATA=get_ItemData()

svd(df_train,ItemData=UseItemData,UserData=UseUserData ,Graph=UseGraphData,lr=LEARNRATE,ureg=UserReg,ireg=ItemReg)
tf.reset_default_graph()
