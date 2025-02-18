import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as req  # Renamed 're' to 'req' to avoid conflicts with 're' (regular expressions)
import matplotlib.pyplot as plt

# Set up Streamlit title and description
st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app is developed for educational purposes. The objective of the app is detecting phishing websites using only content data (not URL-based features). '
         'You can see the details of the approach, dataset, and feature set by clicking on _"See The Details"_. ')

# Project Details Section
with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used _supervised learning_ to classify phishing and legitimate websites, focusing on the HTML content of the websites. '
             'I utilized scikit-learn for the ML models, and features were extracted using BeautifulSoup and the requests library.')
    st.write('For this project, I created my own dataset, which includes features both from the literature and manual analysis.')

    

    st.subheader('Dataset')
    st.write('Data was sourced from _"phishtank.org"_ & _"tranco-list.eu"_.')
    st.write('Total 26,584 websites: **_16,060_ legitimate** websites | **_10,524_ phishing** websites.')
    st.write('The dataset was created in October 2022.')

    # Pie chart visualization
    labels = ['phishing', 'legitimate']
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)  # Exploding the 'phishing' slice

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is drawn as a circle.
    st.pyplot(fig)

    # Display dataframe sample with a slider
    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('Label: 1 for phishing, 0 for legitimate')
    number = st.slider("Select number of rows to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))

    # CSV download button for the dataset
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    # Features and Results section
    st.subheader('Features')
    st.write('This project uses only content-based features extracted from the HTML structure using BeautifulSoup. No URL-based features (e.g., URL length) were used.')

    st.subheader('Results')
    st.write('Seven ML classifiers were tested using k-fold cross-validation. Accuracy, precision, and recall were calculated based on confusion matrices.')
    st.table(ml.df_results)
    st.write('ML Classifiers: NB (Gaussian Naive Bayes), SVM (Support Vector Machine), DT (Decision Tree), RF (Random Forest), AB (AdaBoost), NN (Neural Network), KN (K-Nearest Neighbours)')

# Example Phishing URLs Section
with st.expander('EXAMPLE PHISHING URLs:'):
    st.write('_https://rtyu38.godaddysites.com/_')
    st.write('_https://karafuru.invite-mint.com/_')
    st.write('_https://defi-ned.top/h5/#/_')
    st.caption('Phishing web pages have short lifecycles, so these examples may become invalid over time.')

# Model Selection and URL Input Section
choice = st.selectbox("Select your machine learning model",
                      [
                          'Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                          'AdaBoost', 'Neural Network', 'K-Nearest Neighbours'
                      ])

# Map the selected model
if choice == 'Gaussian Naive Bayes':
    model = ml.nb_model
    st.write('Gaussian Naive Bayes model is selected!')
elif choice == 'Support Vector Machine':
    model = ml.svm_model
    st.write('Support Vector Machine model is selected!')
elif choice == 'Decision Tree':
    model = ml.dt_model
    st.write('Decision Tree model is selected!')
elif choice == 'Random Forest':
    model = ml.rf_model
    st.write('Random Forest model is selected!')
elif choice == 'AdaBoost':
    model = ml.ab_model
    st.write('AdaBoost model is selected!')
elif choice == 'Neural Network':
    model = ml.nn_model
    st.write('Neural Network model is selected!')
else:
    model = ml.kn_model
    st.write('K-Nearest Neighbours model is selected!')

# URL Input and Detection
url = st.text_input('Enter the URL for Phishing Detection')

if st.button('Check!'):
    try:
        response = req.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            st.error(f"HTTP connection was not successful for the URL: {url}")
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]  # Ensures 2D array for prediction
            result = model.predict(vector)
            if result[0] == 0:
                st.success("This web page appears to be legitimate!")
                st.balloons()
            else:
                st.warning("Attention! This web page is a potential PHISHING site!")
                st.snow()

    except req.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
