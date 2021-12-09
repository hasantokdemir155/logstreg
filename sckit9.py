import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


veri3=pd.read_csv('magic041.csv',sep=',')
#print(veri3)
vr3=pd.DataFrame(veri3)


lrg=LogisticRegression()
lb= LabelEncoder()
vr3.Class=lb.fit_transform(vr3.Class)


print(vr3.corr())


y=vr3['Class']
x=vr3.drop('Class',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=123)

lrg.fit(x_train,y_train)

print(lrg.predict(x_test))
m=lrg.predict(x_test)
print(accuracy_score(y_true=y_test,y_pred=m))



