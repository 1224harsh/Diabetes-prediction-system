import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle




df=pd.read_csv('pima-data.csv')
diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)



df['diabetes'] = df['diabetes'].map(diabetes_map)

X=df[['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age','skin']]
y=df['diabetes']
y=y.astype('int')
X=X.astype('int')




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(predictions)

a=[]
for i in range(9):
    j = int(input("enter:"))
    a.append(j)
a_predict=rfc.predict(np.array(a).reshape(1,9))
print(a_predict)

predict_train_data = rfc.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))
pickle.dump(rfc,open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl','rb'))
