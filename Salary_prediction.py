# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 15:31:51 2014
@author: Venkatesh Pappakrishnan
@email: pappakrishnan.venkatesh@gmail.com
"""

import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

class predict_salary():

    def __init__(self):
        print('At Constructor')
        #reading all the input files
        train_features = pd.read_csv("C:/Users/Venkatesh/...")
        test_features = pd.read_csv("C:/Users/Venkatesh/...")
        train_salaries = pd.read_csv("C:/Users/Venkatesh/...")
        
        #identifying the unique list of jobtype, degrees, and majors
        uniquelist = self.unique_list(train_features)
        self.degree_unique = uniquelist[1]
        self.major_unique = uniquelist[2]

        #checking for unclean column and if present proceeding with data cleaning
        unclean_cols_train = self.check_clean(train_features)
        if len(unclean_cols_train)!=0:
            self.check_prob(train_features)
            cleaned_train = self.clean_exec(train_features)
        else:
            cleaned_train = train_features
            
        cleaned_train.to_csv('C:/Users/Venkatesh/...', index = False)    
            
        unclean_cols_test = self.check_clean(test_features)
        if len(unclean_cols_test)!=0:
            cleaned_test = self.clean_exec(test_features)
        else:
            cleaned_test = test_features
        cleaned_test.to_csv('C:/Users/Venkatesh/...', index = False)
             
        #After data cleaning, random forest algorithm is called   
        self.randomforest(cleaned_train, cleaned_test, train_salaries)   
   
    def check_prob(self, train_features): #function to check the probability of occurence of the degree and the major
        self.check_prob_degree(train_features)
        self.check_prob_major(train_features)
            
    def clean_exec(self, dataset): #function to check whether the data is clean or not
        cleaned_degree = self.cleaning_degree(dataset)
        cleaned_features = self.cleaning_major(cleaned_degree) 
        return cleaned_features
        
    def check_clean(self, features): #function to check whether the data is clean or not
        col =[]        
        for column in list(features):
            if 'NONE' in features[column].unique():
                print(column,"- Column is not clean")
                col.append(column)
        return col
        
    def unique_list(self, train_features):  #function to identify the unique list of the degree and the major
        jobtype = train_features['jobType']
        degree = train_features['degree']
        major = train_features['major']

        jobtype_uni = jobtype.unique()   #to identify unique job types
        degree_uni = list(degree.unique())   #to identify unique degree
        degree_uni.remove('NONE')
        major_uni = list(major.unique())   #to identify unique major
        major_uni.remove('NONE')
        return jobtype_uni, degree_uni, major_uni

    def check_prob_degree(self,train_features):
        """group by over single column to get the frequency of each parameters
        #Applying mode type replacement for NONE value cannot be used as the values are equally distributed
        #Calculating the probability of a certain degree for a job type"""
        unique = train_features.loc[:,('jobType', 'degree')]
        unique_group = unique.groupby(['jobType'])
        
        sum_degree = []; jobtype_label = []; degree_label = []; count_label = []
        for name, group in unique_group:
            sum_degree.extend([group['degree'].value_counts().sum()]*len(group['degree'].unique()))

        unique_group2 = unique.groupby(['jobType','degree'])
        for name, group in unique_group2:
            jobtype_label.append(name[0])
            degree_label.append(name[1])
            count_label.append(len(group))

        f = {'jobType': jobtype_label, 'degree': degree_label, 'count':count_label, 'sum':sum_degree}    
        probability_degree = pd.DataFrame(data = f, columns = ['jobType','degree','count','sum'])
        probability_degree['prob'] = probability_degree['count'] /probability_degree['sum']
        """ Observation: Janitor - high school - 50% and None - 50% 
        for other job types - the degrees are distributed uniformly."""

    def cleaning_degree(self, features):
        #replace NONE in degree column
        print ("Replacing NONE in degree column:")
        for i in range(len(features)):
            if features['degree'].ix[i] == 'NONE':
                if features['jobType'].ix[i] != 'JANITOR':
                    features['degree'].ix[i] = random.choice(self.degree_unique) #takes long time
                else:
                    features['degree'].ix[i] = 'HIGH_SCHOOL'
            
            if i/len(features) in (0.25, 0.5, 0.75, 0.95): #To check the progress of the execution
                print((i*100)/len(features),'% completed')

        if 'NONE' in features['degree'].unique():
            print("Degree column is not clean yet")
        else:
            print("Degree column has been cleaned successfully")
            
        return features

    def check_prob_major(self, train_features):
    #Calculating the probability of a major for a job type and degree
        sum_major = []; major_label = []; jobtype_label = []; degree_label = []; count_label = []
        unique = train_features.loc[:,('jobType', 'degree', 'major')]
        unique_group = unique.groupby(['jobType','degree'])

        for name, group in unique_group:
            sum_major.extend([group['major'].value_counts().sum()]*len(group['major'].unique()))

        unique_group2 = unique.groupby(['jobType','degree', 'major'])
        for name, group in unique_group2:
            jobtype_label.append(name[0])
            degree_label.append(name[1])
            major_label.append(name[2])
            count_label.append(len(group))

        d = {'jobType': jobtype_label, 'degree': degree_label, 'major':major_label, 'count':count_label, 'sum':sum_major}    
        probability_major = pd.DataFrame(data = d, columns = ['jobType','degree','major','count','sum'])
        probability_major['prob'] = probability_major['count']/probability_major['sum']
        """ Other than the NONE value in the major column, all the other majors are equally distributed.
        Probability can be used to assign weightage to a particular major but for simplicity. 
        However, for simplicity, we assume all the major are equally distributed since the probabilities are approximately the same."""

    def cleaning_major(self, features):
        # cleaning up major column
        # replace NONE in degree column
        print ("Replacing NONE in major column: ")
        for i in range(len(features)):
            if features['major'].ix[i] == 'NONE':
                if features['major'].ix[i] != 'HIGH_SCHOOL':
                    features['major'].ix[i] = random.choice(self.major_unique) #No major for High school
                    
            if i/len(features) in (0.25, 0.5, 0.75, 0.95):  #To check the progress of the execution
                print((i*100)/len(features),'% completed')

        print("Degree column has been cleaned successfully")
        return features
        
    def randomforest(self, train_features_cleaned, test_features_cleaned, train_salaries):
        print('Implementing Random Forest...')

        jobId_data = pd.Series(train_salaries['jobId'])

        # removing unwanted columns
        del train_features_cleaned['jobId']
        del test_features_cleaned['jobId']
        del train_salaries['jobId']

        le = preprocessing.LabelEncoder()   
        for column in [x for x in list(train_features_cleaned) if x not in ('yearsExperience','milesFromMetropolis')]:
            to_fit = train_features_cleaned[column].unique()
            le.fit(to_fit)
            train_features_cleaned[column] = le.transform(train_features_cleaned[column])
            test_features_cleaned[column] = le.transform(test_features_cleaned[column])

        train_salaries = np.ravel(train_salaries)

        #removing the fields which are less important in salary prediction/calculation
        del train_features_cleaned['major'] #used after calculating the importance factor by considering all features
        del train_features_cleaned['degree']
        del test_features_cleaned['major']
        del test_features_cleaned['degree']

        rfr = RandomForestRegressor(n_estimators = 200, n_jobs = -1, oob_score=True, min_density = 0.05, max_depth = None, min_samples_split = 1)
        salary_fit = rfr.fit(train_features_cleaned, train_salaries)

        fi = enumerate(rfr.feature_importances_)  #provides the importance of every feature
        cols = train_features_cleaned.columns     #if any of the feature is of less importance it can be removed from the Randomforest training
        print([(value,cols[i]) for (i,value) in fi])

        score = salary_fit.score(train_features_cleaned,train_salaries) #from the fit it calculates the score for the existing dataset
        print("Score:", score)

        salary_pred = salary_fit.predict(test_features_cleaned) #predicting the salary using the fit

        salary_pred = pd.Series(salary_pred)

        s = {'jobId': jobId_data, 'salary': salary_pred}    
        salary_predict = pd.DataFrame(data = s, columns = ['jobId', 'salary'])

        salary_predict.to_csv('C:/Users/Venkatesh/...', index=False)


