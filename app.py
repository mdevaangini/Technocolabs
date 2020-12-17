import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(max_df=0.7,stop_words='english')


app=Flask(__name__, template_folder='templates')
model=pickle.load(open('model.pkl', 'rb'))

def get_input(inp):
    df= pd.DataFrame()
    df['comment_text'] = inp
    df['text'] = df['comment_text']
    df.drop('comment_text', axis=1, inplace=True)
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_text = [str(x) for x in request.form.values()]
    df=get_input(inp_text)
    outputs = model.predict_proba(df['text'].tolist())
    
    d={}
    d['toxic']=outputs[0][0]
    d['severe toxic']=outputs[0][1]
    d['obscene']=outputs[0][2]
    d['threat']=outputs[0][3]
    d['insult']=outputs[0][4]
    d['identity hate']=outputs[0][5]
    

    return render_template('index.html', prediction_probabilities='Prediction probabilities are {}'.format(d))


if __name__ == "__main__":
    app.run(threaded=False,port=5000)
