# -*- coding: utf-8 -*-
"""
DSCI 553 Competition Project
@author: Leon Luc
3/23/2022

Method Description:
For the competition project, I have chosen to use an enhanced version of my weighted hybrid recommendation system from HW3. The weighted system incorporates an item-based collaborative filtering recommendation system at 10% weight and a model-based recommendation system using XGBoost at 90% weight. Even though it is highly weighted towards XGBoost, I have found that the 10% from item-based CF helps slightly improve the RMSE compared to not having it at all. To improve the RMSE, I made enhancements to both systems individually. For the model-based system, I added multiple features from the various yelp data sets to the model such as the elite years and age of the user’s account, and the number of check-ins a business has had. I also performed a grid search to find the best values to use for n_estimators and max_depth for XGBoost. For the item-based CF system, I decided to require that the number of co-users shared between two businesses must be 10 or more in order to compute the Pearson Correlation, or else use the average score of the business and user. For both, I made sure to cap the predicted scores between 1 and 5. As far as finding the best weight distribution, I went from using around 75% model-based and 25% item-based CF to the aforementioned 90% model-based and 10% item-based CF. These changes contributed to the improvements of my system as the results below show.

Error Distribution:
>=0 and <1: 101746
>=1 and <2: 21180
>=2 and <3: 7978
>=3 and <4: 4142
>=4: 6998

RMSE:
0.9804936186751556

Execution Time:
262.0783348083496
"""

#import packages
from pyspark import SparkContext
import sys
#import os
import time
import csv
import json
import xgboost as xgb
import numpy as np
import pandas as pd
import math
from datetime import datetime

#initialize Spark Session
sc = SparkContext('local[*]','HW3-task2_3').getOrCreate()
sc.setLogLevel('WARN')

def itemCF(review): #performs item-based CF
    #set reused values
    input_business, input_user = review[0], review[1]
    return_input = (input_user, input_business)

    pearsonweight_and_originalscore = list() #store the Pearson weight (w_i,n) and corresponding original score (r_u,n)
  
    #cover special cases first
    #if the user has no history, use the average score from all users
    user_history = user_reviewed_scores_dict.get(input_user, 'NA')
    if user_history == 'NA':
        return tuple(list(return_input) + [global_user_average,0])
    
    else:
        user_average = np.mean([user_score[1] for user_score in user_history])

    #if the business has no history/ratings, use the average score from the input user
    business_history_self = self_business_scores_dict.get(input_business, 'NA')
    if business_history_self == 'NA':
        return tuple(list(return_input) + [user_average,0])
    
    else:
        #calculate centered (original-avg) score for users who reviewed the business
        business_average = np.mean([*business_history_self.values()]) #get the mean of the input test data business from all the users

    #go through each business reviewed by the user
    for reviewed_business in user_history:
        reviewed_business_id, reviewed_business_score = reviewed_business[0], reviewed_business[1]

        #get the original and centered scores for the reviewed business
        near_business_self = self_business_scores_dict.get(reviewed_business_id)

        #get the users who reviewed both businesses
        co_users = business_history_self.keys() & near_business_self.keys() #intersection
        #print(co_users)
        
        if not co_users or len(co_users) < 10: #if no user both reviewed the same businesses, use the average score from the input business
            return tuple(list(return_input) + [np.mean([user_average,business_average]),0])
        
        #compute co-rated items
        business_history_self2 = {key: business_history_self[key] for key in co_users}
        #print(business_history_self2)
        business_average2 = np.mean([*business_history_self2.values()]) #get the mean of the input test data business from all the users\
        business_history_centered = {user_id: (float(business_history_self2.get(user_id)) - business_average2) for user_id in business_history_self2}
        near_business_self2 = {key: near_business_self[key] for key in co_users}
        #print(near_business_self2)
        near_business_avg = np.mean([*near_business_self2.values()])
        near_business_centered = {user_id: (float(near_business_self2.get(user_id)) - near_business_avg) for user_id in near_business_self2}

        #compute Pearson Correlation weights
        pearsonweight_numerator = list() #(r_u,i-r_ibar)(r_u,j-r_jbar)
        business_history_denominator = list() #sqrt((r_u,i-r_ibar)^2))
        near_business_denominator = list() #sqrt((r_u,j-r_jbar)^2)
        for user in co_users: #get the weight for each co-user
            pearsonweight_numerator = pearsonweight_numerator + [business_history_centered.get(user)*near_business_centered.get(user)]
            business_history_denominator = business_history_denominator + [business_history_centered.get(user)**2]
            near_business_denominator = near_business_denominator + [near_business_centered.get(user)**2]

            if (math.sqrt(sum(business_history_denominator))) * (math.sqrt(sum(near_business_denominator))) != 0:
                pearson_weight = sum(pearsonweight_numerator) / (math.sqrt(sum(business_history_denominator)) * math.sqrt(sum(near_business_denominator)))
                pearsonweight_and_originalscore = pearsonweight_and_originalscore + [(pearson_weight,reviewed_business_score)]
            
            else: #if the denominator of the weight is 0, just set weight to 0
                pearsonweight_and_originalscore = pearsonweight_and_originalscore + [(0,reviewed_business_score)]

    #sort the weights in descending order
    pearsonweight_and_originalscore.sort(key=lambda x: x[0], reverse=True)

    #compute Pearson Similarity using |N|=7
    pearsonsimilarity_numerator = list() #(r_u,n*w_i,n)
    pearsonsimilarity_denominator = list() #|w_i,n|
    for business in pearsonweight_and_originalscore[:7]:
        pearsonsimilarity_numerator = pearsonsimilarity_numerator + [business[0]*business[1]]
        pearsonsimilarity_denominator = pearsonsimilarity_denominator + [math.fabs(business[0])]
    
    #print(pearsonweight_and_originalscore[:7])
    if sum(pearsonsimilarity_denominator) != 0:
        pearson_similarity = sum(pearsonsimilarity_numerator)/sum(pearsonsimilarity_denominator)
    
    else: #for safety, if Pearson similarity denominator is 0, then use the average score from the input business
        return tuple(list(return_input) + [np.mean([user_average,business_average]),len(co_users)])

    return tuple(list(return_input) + [pearson_similarity])

start = time.time()

#load in review data
#reviews_train = sc.textFile('../resource/asnlib/publicdata/yelp_train.csv')
reviews_train = sc.textFile(sys.argv[1] + 'yelp_train.csv')
#reviews_validation = sc.textFile('../resource/asnlib/publicdata/yelp_val_in.csv') #### grading will use yelp_val_in not yelp_val
#reviews_validation = sc.textFile(sys.argv[1] + sys.argv[2]) #folder path and output path
reviews_validation = sc.textFile(sys.argv[2]) #output path

reviews_train = reviews_train.mapPartitions(lambda x: csv.reader(x))
header_reviews_train = reviews_train.first() #remove headers
reviews_train = reviews_train.filter(lambda x: x != header_reviews_train)
#reviews_train.take(10)

reviews_validation = reviews_validation.mapPartitions(lambda x: csv.reader(x))
header_reviews_validation = reviews_validation.first() #remove headers
reviews_validation = reviews_validation.filter(lambda x: x != header_reviews_validation)
#reviews_validation.take(10)

#make final train and validation sets from review
reviews_train_final = reviews_train.map(lambda x: (x[1], (x[0], float(x[2][0])))) #business id, (user id, score) -- need like this to join to business_json first
reviews_validation_final = reviews_validation.map(lambda x: (x[1], x[0])) #don't need stars/scores for validation set
#reviews_train_final.take(5) #(business id, (user id, score))


################Task 2_2: XGBoost
#load in additional data
business_json = sc.textFile(sys.argv[1] + 'business.json').map(lambda x: json.loads(x))
user_json = sc.textFile(sys.argv[1] + 'user.json').map(lambda x: json.loads(x))
checkin = sc.textFile(sys.argv[1] + 'checkin.json').map(lambda x: json.loads(x))
#photo = sc.textFile(sys.argv[1] + 'photo.json').map(lambda x: json.loads(x))
#tip = sc.textFile(sys.argv[1] + 'tip.json').map(lambda x: json.loads(x))

#get features from business (stars and review count) and user (average stars and review count)
business_json_feat = business_json.map(lambda x: (x['business_id'], (x['stars'], x['review_count'], x['is_open'])))
user_json_feat = user_json.map(lambda x: (x["user_id"], (x['average_stars'], x['review_count'], x['useful']+x['funny']+x['cool'],  x['elite'].count('20'), x['friends'].count('20')+1, (datetime.strptime('2022-01-01','%Y-%m-%d') - datetime.strptime(x['yelping_since'],'%Y-%m-%d')).days/365.25, x['compliment_funny'] )))
checkin_feat = checkin.map(lambda x: (x['business_id'], sum(list(x['time'].values()))))
##min year=2004, max year=2018, so using '20' is fine and won't double count 2020


train_feat = reviews_train_final.leftOuterJoin(business_json_feat) # business id, (user id, score), (stars, review_count)  #leftOuterJoin the same count
train_feat = train_feat.map(lambda x: (x[1][0][0], (x[0],x[1][0][1],x[1][1][0],x[1][1][1],x[1][1][2]))) #user id, (business_id, score), (stars, review_count) #swap user id to be the key with business id
train_feat = train_feat.leftOuterJoin(user_json_feat) #user id, (business_id, score, business: stars, review_count), (users: average_stars, review_count, useful)
train_feat = train_feat.map(lambda x: (x[1][0][0], (x[0], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][0][1]))) #x[1][1][3]
#train_feat = train_feat.collect() #business id, (user id, business: stars, review_count, users: average_stars, review_count, useful, score)
train_feat = train_feat.leftOuterJoin(checkin_feat)
train_feat = train_feat.map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10], x[1][1], x[1][0][11])) #x[1][1][3]
train_feat = train_feat.collect() #business id, user id, business: stars, review_count, users: average_stars, review_count, useful, score, checkin

train_feat_df = pd.DataFrame(train_feat, columns = ['business_id','user_id','business_stars','business_review_count','business_is_open','user_average_stars','user_review_count','useful','elite','friends','yelping_since','compliment_funny','checkin','score'])

#if a row has missing values, replace with the mean of the column
train_feat_df['business_stars'].fillna(value=train_feat_df['business_stars'].mean(), inplace=True)
train_feat_df['business_review_count'].fillna(value=train_feat_df['business_review_count'].mean(), inplace=True)
train_feat_df['business_is_open'].fillna(value=train_feat_df['business_is_open'].mean(), inplace=True)
train_feat_df['user_average_stars'].fillna(value=train_feat_df['user_average_stars'].mean(), inplace=True)
train_feat_df['user_review_count'].fillna(value=train_feat_df['user_review_count'].mean(), inplace=True)
train_feat_df['useful'].fillna(value=train_feat_df['useful'].mean(), inplace=True)
train_feat_df['elite'].fillna(value=train_feat_df['elite'].mean(), inplace=True)
train_feat_df['friends'].fillna(value=train_feat_df['friends'].mean(), inplace=True)
train_feat_df['yelping_since'].fillna(value=train_feat_df['yelping_since'].mean(), inplace=True)
train_feat_df['compliment_funny'].fillna(value=train_feat_df['compliment_funny'].mean(), inplace=True)
train_feat_df['checkin'].fillna(value=train_feat_df['checkin'].mean(), inplace=True)
#train_feat_df.head()


#join selected features to validation data
#reviews_train_final: business id, user id
validation_feat = reviews_validation_final.leftOuterJoin(business_json_feat) # business id, (user id, (stars, review_count))  #leftOuterJoin the same count
validation_feat = validation_feat.map(lambda x: (x[1][0], (x[0],x[1][1][0],x[1][1][1],x[1][1][2]))) # user id, (business id, stars, review_count)  ######leftOuterJoin the same count
validation_feat = validation_feat.leftOuterJoin(user_json_feat) #user id, (business_id, business: stars, review_count), (users: average_stars, review_count, useful)
validation_feat = validation_feat.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6]))) #x[1][1][3]
validation_feat = validation_feat.leftOuterJoin(checkin_feat)
validation_feat = validation_feat.map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10], x[1][1])) #x[1][1][3]
validation_feat = validation_feat.collect() #business id, user id, business: stars, review_count, users: average_stars, review_count, useful, checkin

validation_feat_df = pd.DataFrame(validation_feat, columns = ['business_id','user_id','business_stars','business_review_count','business_is_open','user_average_stars','user_review_count','useful','elite','friends','yelping_since','compliment_funny','checkin'])
validation_feat_df = validation_feat_df.sort_values(['user_id','business_id'])

#if a row has missing values, replace with the mean of the column
validation_feat_df['business_stars'].fillna(value=validation_feat_df['business_stars'].mean(), inplace=True)
validation_feat_df['business_review_count'].fillna(value=validation_feat_df['business_review_count'].mean(), inplace=True)
validation_feat_df['business_is_open'].fillna(value=validation_feat_df['business_is_open'].mean(), inplace=True)
validation_feat_df['user_average_stars'].fillna(value=validation_feat_df['user_average_stars'].mean(), inplace=True)
validation_feat_df['user_review_count'].fillna(value=validation_feat_df['user_review_count'].mean(), inplace=True)
validation_feat_df['useful'].fillna(value=validation_feat_df['useful'].mean(), inplace=True)
validation_feat_df['elite'].fillna(value=validation_feat_df['elite'].mean(), inplace=True)
validation_feat_df['friends'].fillna(value=validation_feat_df['friends'].mean(), inplace=True)
validation_feat_df['yelping_since'].fillna(value=validation_feat_df['yelping_since'].mean(), inplace=True)
validation_feat_df['compliment_funny'].fillna(value=validation_feat_df['compliment_funny'].mean(), inplace=True)
validation_feat_df['checkin'].fillna(value=validation_feat_df['checkin'].mean(), inplace=True)
#validation_feat_df.head()

#separate into train, test (validation) sets for modeling
X_train = train_feat_df.drop(['business_id','user_id','score'], axis=1)
#xgb_train = xgb.DMatrix(X_train, label = y_train)

y_train = train_feat_df['score']

X_test = validation_feat_df.drop(['business_id','user_id',], axis=1)
#xgb_test = xgb.DMatrix(X_test)

#train xgboost model
xgb_mod = xgb.XGBRegressor(n_estimators=600, max_depth=5, random_state=100)
xgb_mod.fit(X_train, y_train)

y_pred = xgb_mod.predict(X_test)

##update predictions so that if pred is <1, set to 1, and if pred >5, set to 5
y_pred2 = y_pred.copy()
y_pred2[y_pred2 < 1] = 1
y_pred2[y_pred2 > 5] = 5

#save results
xgboost_pred = reviews_validation.map(lambda x: [x[0], x[1]])
#xgboost_pred = xgboost_pred.collect()
xgboost_pred = xgboost_pred.sortBy(lambda x: (x[0],x[1])).collect() #need to sort by user id, business id to match X_train order
for i in range(len(xgboost_pred)):
    xgboost_pred[i].append(float(y_pred2[i])) #add predicted score to test/validation rows

        
################Task 2_1: Item-based CF
#get all scores made by users for each business
self_business_scores = reviews_train_final.groupByKey().mapValues(list).collect()
self_business_scores_dict = dict([(self_business_scores[index][0], dict(self_business_scores[index][1])) for index in range(len(self_business_scores))])

#get all scores each user made for a business
user_reviewed_scores = reviews_train_final.map(lambda x: (x[1][0], (x[0],x[1][1]))).groupByKey().mapValues(list).collect()
user_reviewed_scores_dict = dict([(user_reviewed_scores[index][0], user_reviewed_scores[index][1]) for index in range(len(user_reviewed_scores))])

#get average score across all users in train set
global_user_average = np.mean(reviews_train_final.map(lambda x: x[1][1]).collect())

#perform item-based CF on validation set
itemCF_pred= reviews_validation_final.map(lambda review: itemCF(review))
#itemCF_pred = itemCF_pred.collect()
itemCF_pred = itemCF_pred.sortBy(lambda x: (x[0],x[1])).collect() #use this if want to compare to task 2_2 results --- don't use this for the submission



######Hybrid approach
#make dataframes for the predictions
itemCF_pred_df = pd.DataFrame(itemCF_pred, columns = ['user_id', 'business_id', 'prediction_itemCF', 'co-users'])
xgboost_pred_df = pd.DataFrame(xgboost_pred, columns = ['user_id', 'business_id', 'prediction_XGBoost'])
#itemCF_pred_df.head()
#xgboost_pred_df.head()

combined_pred_df = itemCF_pred_df.copy()
combined_pred_df['prediction_XGBoost'] = xgboost_pred_df['prediction_XGBoost']
combined_pred_df['business_review_count'] = validation_feat_df['business_review_count']
#combined_pred_df.head()

#can write if-else rules to decide on weights of each prediction here
combined_pred_df['prediction'] = 0
#combined_pred_df.loc[( (combined_pred_df['co-users'] < 10) ),'prediction'] = combined_pred_df['prediction_XGBoost']
#combined_pred_df.loc[( (combined_pred_df['co-users'] >= 10) ),'prediction'] = combined_pred_df['prediction_itemCF']
#combined_pred_df.loc[( (combined_pred_df['business_review_count'] >= 100) ),'prediction'] = (0.1*combined_pred_df['prediction_itemCF'])+(0.9*combined_pred_df['prediction_XGBoost'])

combined_pred_df['prediction'] = (0.1*combined_pred_df['prediction_itemCF'])+(0.9*combined_pred_df['prediction_XGBoost'])
#combined_pred_df.loc[( (combined_pred_df['prediction'] <= 2.75) & (combined_pred_df['prediction'] >= 2.5) ),'prediction'] = 2
#combined_pred_df.head()


combined_predicted = list()
#for i in range(len(itemCF_pred)):
    ##combined_predicted.append((itemCF_pred[i][0], itemCF_pred[i][1], itemCF_pred[i][2], xgboost_pred[i][2]))
    #combined_predicted.append((itemCF_pred[i][0], itemCF_pred[i][1], (0.1*itemCF_pred[i][2])+(0.9*xgboost_pred[i][2])))

for i in range(len(combined_pred_df)):
    #combined_predicted.append((itemCF_pred[i][0], itemCF_pred[i][1], itemCF_pred[i][2], xgboost_pred[i][2]))
    combined_predicted.append((combined_pred_df['user_id'][i], combined_pred_df['business_id'][i], combined_pred_df['prediction'][i]))

#with open('./results_new.csv', 'w') as fp:
with open(sys.argv[3], 'w') as fp: #result path
    fp.write('user_id, business_id, prediction\n') #add headers
    for row in combined_predicted:
        fp.write(f'{row[0]},{row[1]},{row[2]}\n') ########change row[2] to float(row[2]) if error in submission    
    
end = time.time()
print(f'Duration: {end-start}')


######to compare predictions to ground truth
#predicted = list()
#for i in range(len(xgboost_pred)):
    #predicted.append((xgboost_pred[i][0], xgboost_pred[i][1], float(xgboost_pred[i][2])))

actual = list()
#with open('../resource/asnlib/publicdata/yelp_val.csv') as fp:
#with open(sys.argv[1] + 'yelp_val.csv') as fp:
with open(sys.argv[2]) as fp:
    act = csv.reader(fp)
    for pair in list(act)[1:]:
        actual.append((pair[0], pair[1], pair[2]))
actual = sorted(actual, key = lambda x: (x[0],x[1])) #sort by user id, business id to match xgboost_pred

#calculate RMSE
differences = list()
for i in range(len(combined_predicted)):
    differences.append((float(combined_predicted[i][2]) - float(actual[i][2]))**2)
print(f'RMSE: {math.sqrt(sum(differences)/len(combined_predicted))}')

############### to store the difference margin (for competition)
diff_margin = {'>=0 and <1': 0, '>=1 and <2': 0, '>=2 and <3': 0, '>=3 and <4': 0, '>=4': 0}
for diff in differences:
  if math.fabs(diff) < 1:
    diff_margin['>=0 and <1'] = diff_margin.get('>=0 and <1')+1
  elif 1 <= math.fabs(diff) < 2:
    diff_margin['>=1 and <2'] = diff_margin.get('>=1 and <2')+1
  elif 2 <= math.fabs(diff) < 3:
    diff_margin['>=2 and <3'] = diff_margin.get('>=2 and <3')+1
  elif 3 <= math.fabs(diff) < 4:
    diff_margin['>=3 and <4'] = diff_margin.get('>=3 and <4')+1
  else:
    diff_margin['>=4'] = diff_margin.get('>=4')+1
print(diff_margin)

#export PYSPARK_PYTHON=python3.6
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit task2_3.py '../resource/asnlib/publicdata/' 'yelp_val_in.csv' './results_task2_2.csv'
