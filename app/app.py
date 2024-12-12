import streamlit as st
import pickle
import utils

model = pickle.load(open('model/rf_model.pkl', 'rb'))

st.header('Duplicate Question Pairs Predictor')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = utils.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')