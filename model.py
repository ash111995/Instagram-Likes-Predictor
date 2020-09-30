
#Import libraries
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"C:\Users\Devashish Barik\Desktop\Ashish\Freelancing Projects\Instagram Likes Prediction\dataset.csv")
data_common = pd.read_csv(r"C:\Users\Devashish Barik\Desktop\Ashish\Freelancing Projects\Instagram Likes Prediction\kind.csv")

data1 = data[['date','username','numberPosts','numberFollowing','numberFollowers','numberLikes','isVideo','tags']]
data2 = data_common[['Time','Username','No. of Post','following','followers','Likes','is Video','Tags']]
data2.rename(columns = {'Time':'date','Username':'username','No. of Post':'numberPosts','following':'numberFollowing','followers':'numberFollowers','Likes':'numberLikes','is Video':'isVideo','Tags':'tags'}, inplace = True)
data1 = pd.concat([data1,data2])



number_tags = []
for i in data1.tags:
    if i == '[]':
        number_tags.append(0)
    else:
        number_tags.append(len(i.strip('[').strip(']').split(',')))
        
data1['number_tags'] = number_tags

df = data1.groupby(['username'])['numberPosts','numberFollowing','numberFollowers','numberLikes','number_tags'].mean()

df.reset_index(drop = True, inplace = True)

df['number_tags'] = round(df.number_tags)

df.rename(columns = {'number_tags':'Avg_number_tags','numberLikes':'Avg_number_likes'},inplace = True)

df["Avg_number_likes"] = np.log1p(df["Avg_number_likes"])

df["numberFollowers"] = np.log1p(df["numberFollowers"])

df["numberPosts"] = np.log1p(df["numberPosts"])

df["numberFollowing"] = np.log1p(df["numberFollowing"])


x= df.drop(['Avg_number_likes','Avg_number_tags'], axis = 1)
y = df.Avg_number_likes

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.1, random_state =1)
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import pickle

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
x_train_pred = clf.predict(x_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))


"""
p = np.log(173)
fg = np.log(862)
fw = np.log(917)
#t = 0

sample = [[p, fg, fw ]]
#sample = sc.transform(sample)

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(np.exp(model.predict(sample)))
"""


