import pandas as pd
import sklearn
fname = 'train.csv'

data = pd.read_csv(fname)


data.head()


from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

data['Sex_numeric']=lb.fit_transform(data['Sex'])

data.head()



x = data[["Pclass","Age","SibSp","Parch","Sex_numeric"]]


y=data[["Survived"]]

x.head()

y.head()




from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=11)

x=x.fillna(0)

knn.fit(x,y)

y_pred = knn.predict(x)

y_pred

sklearn.metrics.accuracy_score(y,y_pred)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x,y)

rf.predict(x) 
