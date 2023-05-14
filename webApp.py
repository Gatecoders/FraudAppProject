import streamlit as st
import joblib

model = joblib.load('./model.pkl')
vectorizer = joblib.load('./vectorizer.pkl')

def predict_review(review_text):
    review_text = [review_text]
    review_text = vectorizer.transform(review_text).toarray()
    prediction = model.predict(review_text)
    return prediction[0]

def app():
    st.title('Review Sentiment Analysis App')

    review = st.text_input('Enter your review:')
    if st.button('Submit'):
        prediction = predict_review(review)
        if prediction == 1:
            st.write('This app is not fraudulent.')
            st.balloons();
        else:
            st.write('This app is fraudulent.')

        positive_proba = model.predict_proba(vectorizer.transform([review]))[0][1]
        negative_proba = model.predict_proba(vectorizer.transform([review]))[0][0]
        st.write(f"Positive sentiment probability: {round(positive_proba*100)}%")
        st.write(f"Negative sentiment probability: {round(negative_proba*100)}%")
        # st.slider("Sentiment Percentage", 0, 100, (positive_proba*100))


if __name__ == '__main__':
    app()
