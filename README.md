# d14
#regression
import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Close']*100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
print(df.head()) =>             Adj. Close    HL_PCT  PCT_change  Adj. Volume
                    Date                                                     
                    2004-08-19   50.322842  3.712563    0.323915   44659000.0
                    2004-08-20   54.322689  0.710922    6.739913   22834300.0
                    2004-08-23   54.869377  3.729433   -1.243144   18256100.0
                    2004-08-24   52.597363  6.417469   -6.074187   15247300.0
                    2004-08-25   53.164113  1.886792    1.169811    9188600.0

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head()) =>             Adj. Close    HL_PCT  PCT_change  Adj. Volume      label
                    Date                                                                
                    2004-08-19   50.322842  3.712563    0.323915   44659000.0  69.078238
                    2004-08-20   54.322689  0.710922    6.739913   22834300.0  67.839414
                    2004-08-23   54.869377  3.729433   -1.243144   18256100.0  68.912727
                    2004-08-24   52.597363  6.417469   -6.074187   15247300.0  70.668146
                    2004-08-25   53.164113  1.886792    1.169811    9188600.0  71.219849
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
#X = X[:-forecast_out+1]
#df.dropna(inplace=True)
#y = np.array(df['label'])
print(len(X),len(y)) => 3424,3424

#K-Means
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11]])
plt.scatter(X[:,0],X[:,1],s=150)
plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
colors = 10*['g.','r.','c.','b.','k.']
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=20)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5)
plt.show()

#http://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xlrd

df = pd.read_excel('titanic.xls')
print(df.head())

def handle_non_numerical_data(df):                                #数据处理
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x +=1
            df[column] = list(map(convert_to_int,df[column]))
    return df
df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived'],1).astype(float))                       #KMeans
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = KMeans(n_clusters=2)
clf.fit(X)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
print(correct/len(X))
