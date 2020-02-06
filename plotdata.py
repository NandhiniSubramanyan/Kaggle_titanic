# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:24:10 2019

@author: RanjaniSubramanyan
"""
# CATEGORISE Age, Fare and then plot. May get some conclusive insights
# import statements
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
#%%
# read input data in csv format
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print("# of passengers in Titanic : {}".format(len(train_data)))
print(train_data.shape, test_data.shape)
print(len(train_data.columns), len(test_data.columns))
#%%
sns.pairplot(train_data, hue="Survived")
#%%
sums = train_data.Survived.groupby(train_data.Sex).sum()
plt.axis('equal')
plt.pie(sums, labels=sums.index)
#%%
sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=train_data)
sns.catplot(x="Survived", y="Fare", hue="Sex", kind="swarm", data=train_data)
sns.catplot(x="Survived", y="Pclass", hue="Sex", kind="swarm", data=train_data)
#%%
female = train_data.loc[(train_data["Sex"]=="female"), ["Sex"]]
print("# of female in Titanic : {}".format(len(female)))
female_below_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]<50), ["Sex","Age"]]
female_above_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]>50), ["Sex","Age"]]
female_age_survival_below_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]<50) & (train_data["Survived"]==1), ["Sex","Age","Survived"]]
female_age_survival_above_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]>50) & (train_data["Survived"]==1), ["Sex","Age","Survived"]]
print("# of female in Titanic below age 50 is {} and {} of them survived".format(len(female_below_50), len(female_age_survival_below_50)))
print("# of female in Titanic above age 50 is {} and {} of them survived".format(len(female_above_50), len(female_age_survival_above_50)))
#%%
"""NaN values finding and filling"""
train_data['title'] = [row.split(',')[1].split('.')[0] for row in train_data.Name]
train_data['Age'] = train_data.groupby(['title', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
test_data['title'] = [row.split(',')[1].split('.')[0] for row in test_data.Name]
test_data['Age'] = test_data.groupby(['title', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
print(test_data[test_data['title'] == ' Ms'].index.tolist()[0])
print(train_data[train_data['title'] == ' Ms'].index.tolist())
age_in_ms_title_train = train_data.iloc[train_data[train_data['title'] == ' Ms'].index.tolist()[0]]['Age']
print(age_in_ms_title_train)
test_data.loc[[test_data[test_data['title'] == ' Ms'].index.tolist()[0]], 'Age'] = age_in_ms_title_train
print(train_data.shape, test_data.shape)
train_data = train_data.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'title'], axis=1)
passengerId = test_data['PassengerId']
test_data = test_data.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'title'], axis=1)
print(train_data.shape, test_data.shape)
train_data = pd.get_dummies(data=train_data, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(data=test_data, columns=['Sex', 'Embarked'])
test_data = test_data.fillna(0)
print(train_data.shape, test_data.shape)
#%%
y = train_data['Survived']
X = train_data.drop(['Survived'], axis=1)
print(X.shape, y.shape)
#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(random_state=0, n_estimators=30, criterion='gini', max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print(accuracy_score(y_val, y_pred))
#%%
y_test_pred = pd.DataFrame(clf.predict(test_data), columns=['Survived'])
#%%
final_submission = pd.concat([passengerId, y_test_pred], axis=1)
print(final_submission.shape)
final_submission.to_csv('submission.csv', sep = ',', index = False, quoting=csv.QUOTE_ALL)