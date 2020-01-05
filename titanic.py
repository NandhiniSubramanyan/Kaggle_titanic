# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:25:51 2019

@author: NandhiniSubramanyan
"""
#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
print('import successful')
sns.set(color_codes=True)
sns.set()

#%%
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print('Shape of train, test data : {} {}'.format(train_data.shape, test_data.shape))

#%%Distribution of sex with survival rate
sums = train_data.Survived.groupby(train_data.Sex).sum()
plt.axis('equal')
plt.title('Distribution of sex with survival')
plt.pie(sums, labels=sums.index)

# Distribution of fare with survival rate
ax = sns.catplot(x="Survived", y="Fare", hue="Sex", data=train_data)
plt.title('Distribution of sex over fare with survival')

ax1 = sns.catplot(x="Survived", y="Age", hue="Sex", data=train_data)
plt.title('Distribution of sex over age with survival')

#%%
train_data['title'] = [row.split(',')[1].split('.')[0] for row in train_data.Name]
train_data['Age'] = train_data.groupby(['title', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
test_data['title'] = [row.split(',')[1].split('.')[0] for row in test_data.Name]
test_data['Age'] = test_data.groupby(['title', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))

#%%
print(test_data.loc[pd.isna(test_data['Age']), :].index)
print(train_data.title.unique())
print(test_data.title.unique())
print(test_data[test_data['title'] == ' Ms'].index)
print(train_data[train_data['title'] == ' Ms'].index.tolist())
print(train_data.iloc[443,:])
print(train_data.isna().sum())

#%%
sums = train_data.Survived.groupby(train_data.Sex).sum()
plt.axis('equal')
plt.pie(sums, labels=sums.index)

#%%
sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=train_data)
sns.catplot(x="Survived", y="Fare", hue="Sex", kind="swarm", data=train_data)
sns.catplot(x="Survived", y="Pclass", hue="Sex", kind="swarm", data=train_data)

#%%
# statistics comparing survival rate in men, women and their age
female = train_data.loc[(train_data["Sex"]=="female"), ["Sex"]]
print("# of female in Titanic : {}".format(len(female)))
female_below_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]<50), ["Sex","Age"]]
female_above_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]>50), ["Sex","Age"]]
female_age_survival_below_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]<50) & (train_data["Survived"]==1), ["Sex","Age","Survived"]]
female_age_survival_above_50 = train_data.loc[(train_data["Sex"]=="female") & (train_data["Age"]>50) & (train_data["Survived"]==1), ["Sex","Age","Survived"]]
print("# of female in Titanic below age 50 is {} and {} of them survived".format(len(female_below_50), len(female_age_survival_below_50)))
print("# of female in Titanic above age 50 is {} and {} of them survived".format(len(female_above_50), len(female_age_survival_above_50)))
