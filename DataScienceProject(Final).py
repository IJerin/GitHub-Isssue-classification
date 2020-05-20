#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 00:35:49 2020

@author: ismotjerin
"""

import sys
!{sys.executable} -m pip install pycoshark
from mongoengine import connect
from pycoshark.mongomodels import People, Commit, Project, VCSSystem,IssueSystem, Issue
from pycoshark.utils import create_mongodb_uri_string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import csv
import numpy as np
import seaborn as sns


# Database credentials
user = something
password = something
host = something
port = something
authentication_db = 'smartshark'
database = "smartshark"
ssl_enabled = None

# Establish connection
uri = create_mongodb_uri_string(user, password, host, port, authentication_db, ssl_enabled)
connect(database, host=uri)

# Fetch project id and version control system id for the 'kafka' project
# The only() decides the data that is actually retrieved from the MongoDB. Always restrict this to the field that you require!

projects = ['ant-ivy', 'archiva', 'calcite', 'cayenne', 'commons-bcel', 'commons-beanutils', 'commons-codec', 'commons-collections',
            'commons-compress', 'commons-configuration', 'commons-dbcp', 'commons-digester', 'commons-io',
            'commons-jcs', 'commons-jexl', 'commons-lang', 'commons-math', 'commons-net', 'commons-rdf', 'commons-scxml']

rows_list = []

for projectName in projects:
    project = Project.objects(name=projectName).only('id').get()
    #vcs_system = VCSSystem.objects(project_id=project.id).only('id','url').get()

    #getting issue id from the project
    issue_id = IssueSystem.objects(project_id=project.id).only('id','url').get()

###########Getting data ready############
  
    for issue in Issue.objects(issue_system_id=issue_id.id).only('issue_type','desc','title','priority', 'status').timeout(False):
       
        for row in issue:
    
            dict1 = {}
            dict1.update({'Id':issue_id.id})
            dict1.update({'Description':issue.desc})
            dict1.update({'Title':issue.title})
            dict1.update({'Issue_Type':issue.issue_type})
        rows_list.append(dict1)
rows_list   
df = pd.DataFrame(rows_list)
df 

df.head(40)
      
            
df.to_csv("projectData.csv",index=False)
data = pd.read_csv("projectData.csv")
data = data[pd.notnull(df['Description'])]
data = data[pd.notnull(df['Title'])]
data.head()
data.loc[df['Issue_Type'] != 'Bug', 'Issue_Type'] = 'Issue'

data['Issue_Type_ID'] = data['Issue_Type'].factorize()[0]
data['Issue_Type_ID']

########## Feature selection ##################

# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
# features = tfidf.fit_transform(data.Title,data.Description).toarray()
# #features = tfidf.fit_transform(data,Title.values.astype('U'))
# labels = data.Issue_Type_ID
# features.shape #(417000, 72763)


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

################### Naive Bayes Model ###################
X_train, X_test, y_train, y_test = train_test_split(data['Title'], data['Issue_Type'], test_size=0.33, random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

lebels = clf.predict(count_vect.transform(X_test))
lebels
conf_mat = confusion_matrix(y_test, lebels)
conf_mat
result = accuracy(conf_mat)
result #0.8499952038369305
sns.heatmap(conf_mat, xticklabels=0, yticklabels=0)
