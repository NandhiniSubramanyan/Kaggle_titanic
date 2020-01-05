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
#%%
# read input data in csv format
train_data = pd.read_csv('train.csv')
print("# of passengers in Titanic : {}".format(len(train_data)))
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