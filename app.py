import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from win32com.client import Dispatch

def speak(text):
	speak=Dispatch(("SAPI.SpVoice"))
	speak.Speak(text)


model = pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))


def main():
	st.title("Email Spam Classification Application")
	st.write("By Hitesh Kumar")
	activites=["Classification","About"]
	choices=st.sidebar.selectbox("Select Activities",activites)
	if choices=="Classification":
		st.subheader("Classification")
		msg=st.text_area("Enter your message")
		if st.button("Predict"):
			print(msg)
			print(type(msg))
			data=[msg]
			print(data)
			vec=cv.transform(data).toarray()
			result=model.predict(vec)
			if result[0]==0:
				st.success("This is Not A Spam Email")
				speak("This is Not A Spam Email")
			else:
				st.error("This is A Spam Email")
				speak("This is A Spam Email")
main()
