import streamlit as st
import Email_extractor
import joblib

model = joblib.load('Spam_detect.pkl')
st.title("This is a demo of a spam email detection model")
user_input = st.chat_input("Enter the email text here")
st.write(user_input)
features = Email_extractor.extractFeatures(str(user_input))
prediction= model.predict(features)
if(user_input):
    if prediction[0]:
        st.warning("SPAM DETECTED!! ❌❌❌")
    if prediction[0]==0:
        st.success("NOT A SPAM")

print(user_input)

