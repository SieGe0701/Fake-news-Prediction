import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import re
import string

pkl_lr = open("lr.pkl","rb")
lr = pickle.load(pkl_lr)
pkl_dt = open("dt.pkl","rb")
dt = pickle.load(pkl_dt)
pkl_rfc = open("rfc.pkl","rb")
rfc = pickle.load(pkl_rfc)
pkl_xgb = open("xgb.pkl","rb")
xgb = pickle.load(pkl_xgb)
pkl_vect = open("vect.pkl","rb")
vectorizer = pickle.load(pkl_vect)

def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('\\W'," ",text)
    text = re.sub("https?://\S+|www\.\S+",'',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text

def prediction(n):
    if n == 0:
        return "Fake news"
    elif n == 1:
        return "True news"

def fake_news_prediction(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(preprocess)
    x_test = new_def_test['text']
    new_xv_test = vectorizer.transform(x_test)
    pred_lr = lr.predict(new_xv_test)
    pred_dt = dt.predict(new_xv_test)
    pred_xgb = xgb.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    return ("\n\n LR Prediction : {} \n XGB Prediction : {} \n DT Prediction : {} \n RFC Prediction : {}".format(prediction(pred_lr[0]),
                                                                                                                      prediction(pred_xgb[0]),
                                                                                                                      prediction(pred_dt[0]),
                                                                                                                      prediction(pred_rfc[0])))

def main():
    st.title(":orange[Fake news prediction]")
    st.divider()
    with st.form(key="news_form",clear_on_submit=True):
        news = st.text_area("News_Input",placeholder="Enter news",height=300,label_visibility="hidden")
        submit_button = st.form_submit_button("Predict")
    result=""
    st.divider()
    col1,col2 = st.columns(2)
    if submit_button:
        result = fake_news_prediction(news)
        st.success(result)
    if col2.button("Reset"):
        news.replace(news,"Paste news here")    
if __name__ == '__main__':
    main()