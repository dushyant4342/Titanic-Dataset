import pandas as pd

fname='train.csv'

data=pd.read_csv(fname)

print(len(data))

print(data.shape)

print(data.head())

print(data.columns)

print(data.count())

print(data['Age'].min(), data['Age'].max())

print(data['Survived'].value_counts())

print(data['Survived'].value_counts()*100/len(data))



print(data['Sex'].value_counts())



print(data['Pclass'].value_counts())



%matplotlib inline
alpha_color=1
data['Survived'].value_counts().plot(kind='bar')


data['Sex'].value_counts().plot(kind='bar',
                                color=['b','r'], alpha=alpha_color)



data['Pclass'].value_counts().sort_index().plot(kind='bar')


data.plot(kind='scatter',x='Survived',y='Age')


data[data['Survived']==1]['Age'].value_counts().sort_index().plot(kind='bar')



bins=[0,10,20,30,40,50,60,70,80]
data['AgeBin']=pd.cut(data['Age'],bins)
data[data['Survived']==1]['AgeBin'].value_counts().sort_index().plot(kind='bar')




data[data['Survived']==0]['AgeBin'].value_counts().sort_index().plot(kind='bar')





data['AgeBin'].value_counts().sort_index().plot(kind='bar')




data[data['Pclass']==1]['Survived'].value_counts().sort_index().plot(kind='bar')





data[data['Pclass']==2]['Survived'].value_counts().sort_index().plot(kind='bar')




data[data['Pclass']==3]['Survived'].value_counts().sort_index().plot(kind='bar')



data[data['Sex']=='male']['Survived'].value_counts().sort_index().plot(kind='bar')



data[data['Sex']=='female']['Survived'].value_counts().sort_index().plot(kind='bar')





data[(data['Sex']=='male')&(data['Pclass']==1)]['Survived'].value_counts().plot(kind='bar')





data[(data['Sex']=='male')&(data['Pclass']==3)]['Survived'].value_counts().sort_index().plot(kind='bar')



data[(data['Sex']=='female') & (data['Pclass']==1)]['Survived'].value_counts().sort_index().plot(kind='bar')


data[(data['Sex']=='female') & (data['Pclass']==3)]['Survived'].value_counts().sort_index().plot(kind='bar')





data[(data['Sex']=='female') & (data['Pclass']==2)]['Survived'].value_counts().sort_index().plot(kind='bar')





from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

data['Sex_numeric']=lb.fit_transform(data['Sex'])

data.head()



x = data[["Pclass","Age","SibSp","Parch","Sex_numeric"]]


y=data[["Survived"]]

x.head()

y.head()



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x,y)

rf.predict(x) 




