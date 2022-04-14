from flask import Flask, render_template, request

app = Flask(__name__)

#----Model-----#

import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


#Load dataset. This data is was downloaded from basketball reference.com
df = pd.read_csv('celtics_nba.csv')

#Make all columns lowercase
df.columns = df.columns.str.lower()

#Drop redundant rk column 
df = df.drop(columns='rk')

#Rename colums
df['three'] = df['c_3p%']
df['c_fgp'] = df['c_fg%']

#Make new column to specify if a game is home or away
df['home_away'] = np.where(df['home_away'].isin(['@']), 0, 1)

#Make date column into date dtype
df['date'] = pd.to_datetime(df['date'])

#Make win loss a 1 or 0 
df['w/l'] = np.where(df['w/l'].isin(['W']), 1, 0)

#Dataframe with the features we will use for our model
df_model = df[['home_away','opp','c_fgp','three','celtics_points','c_ast','c_trb']]



#ML Model

#Defining X, y

X = df_model
y = df['w/l']

#Train Test Split on data

X_train, X_test, y_train, y_test = train_test_split(X, y)

#Instantiate standard scaler and OneHotEnocder

scaler = StandardScaler(with_mean=False)
enc = ColumnTransformer(transformers =[
    ('enc', OneHotEncoder(sparse = False, handle_unknown = 'ignore'), list(range(2))),
], remainder ='passthrough')

#Instantiate Model

forest = RandomForestClassifier(criterion='entropy', max_depth=3)

#Create a model pipeline, create a voting classifier that weight more on RandomForestClassifier 

pipe = Pipeline(steps =[
    ('Encoder', enc),
    ('Scaler', scaler),
    ('model', forest)])

pipe.fit(X_train, y_train)

pd.to_pickle(pipe,r'/Users/nmsievers/Desktop/Celtics/celtics_dtcmodel')
model = pd.read_pickle(r'/Users/nmsievers/Desktop/Celtics/celtics_dtcmodel')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('test.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    home_away = request.form['home_away']
    opp = request.form['opp']
    c_fgp = request.form['c_fgp']
    three = request.form['three']
    celtics_points = request.form['celtics_points']
    c_ast = request.form['c_ast']
    c_trb = request.form['c_trb']
    item = ([[home_away, opp, c_fgp, three ,celtics_points,c_ast,c_trb]])
    score = model.predict(item) 
    if score == 1:
        return render_template('test.html',pred='The model predicts a Celtics Win!')
    else:
        return render_template('test.html', pred='The model predicts a Celtics Loss.')


if __name__ == '__main__':
    app.run(port=3000,debug=True)
