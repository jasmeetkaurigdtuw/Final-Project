# pip install streamlit
# pip install flask
# pip install flask_sqlalchemy
# pip install flask_migrate
# pip install scikit-learn
# pip install pandas
# pip install --upgrade streamlit
# pip install matplotlib
# pip install openai
# pip install openai==0.28
# pip install psycopg2

# activate venv (virtual environment): source ./venv/Scripts/activate
# right-click app.py and open in integrated terminal
# run app by running: streamlit run app.py
#------------------------------------------------------------------------------------------------------------
# Importing required libraries and packages
import openai
import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Reading the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Checking if the API key is provided
if api_key is None:
    raise ValueError("OpenAI API key is not provided. Set the environment variable 'OPENAI_API_KEY'.")

# Setting the API key for the OpenAI module
openai.api_key = api_key

# Reading data from ElephantSQL database URI
database_uri = 'postgresql://caqkfcsy:b41wIKaMBjOtBNDe1Y6EL_Xz-uoGqvuU@salt.db.elephantsql.com/caqkfcsy'

# Creating the database engine
engine = create_engine(database_uri)

# Loading the dataset from PostgreSQL
heart_data = pd.read_sql('SELECT * FROM heart_data', engine)

# Renaming columns based on PostgreSQL column names
heart_data.columns = ['age', 'sex', 'chestpaintype', 'restingbp', 'cholesterol', 'fastingbs', 'restingecg', 'maxhr', 'exerciseangina', 'oldpeak', 'st_slope', 'heartdisease']

# Mapping for categorical features
sex_mapping = {'M': 1, 'F': 0}
cp_mapping = {'NAP': 0, 'ATA': 1, 'ASY': 2, 'TA': 3}
bs_mapping = {'Normal': 0, 'ST': 1}
ecg_mapping = {'Normal': 0, 'ST': 1}
angina_mapping = {'N': 0, 'Y': 1}
slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

# Converting categorical columns to numeric in the original dataset
heart_data['sex'] = heart_data['sex'].map(sex_mapping)
heart_data['chestpaintype'] = heart_data['chestpaintype'].map(cp_mapping)
heart_data['fastingbs'] = heart_data['fastingbs'].map(bs_mapping)
heart_data['restingecg'] = heart_data['restingecg'].map(ecg_mapping)
heart_data['exerciseangina'] = heart_data['exerciseangina'].map(angina_mapping)
heart_data['st_slope'] = heart_data['st_slope'].map(slope_mapping)

# Rendering the header
st.markdown("""
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CardioGuard AI: Predicting Heart Disease & AI Lifestyle Advice</title>
    <link rel="stylesheet" type="text/css" href="static/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
</head>
<header class="header">
    <h1>CardioGuard AI</h1>
    <p>Predicting Heart Disease Risk & Providing Lifestyle Advice using AI</p>
</header>
""", unsafe_allow_html=True)

# Rendering the content
st.markdown("""
<div class="container">
    <div class="content">
        <h2 class="shining-text">Welcome to CardioGuard AI: Your Personalized Health Assistant!</h2>
        <p>Empower yourself with personalized predictions for heart disease risk and actionable lifestyle advice, all powered by cutting-edge AI technology.</p>
        <p>To get started, simply enter your information in the sidebar, and let us guide you through insightful predictions and tailored recommendations.</p>
        <p>Don't forget to engage with our friendly chatbot, Hearty, who's here to answer your questions and provide additional support along your health journey!</p>
    </div>
</div>
""", unsafe_allow_html=True)


# Adding HealthyHeart image to the header 
healthy_heart_image = "Resources/HealthyHeart.png"
openAI_logo = "Resources/OpenAI.png"
chatgpt_logo = "Resources/ChatGPT.png"
chatbot_logo = "Resources/Chatbot.png"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(healthy_heart_image, caption="Healthy Hearts with CardioGuard AI", use_column_width=False, width=100)
with col2:
    st.image(openAI_logo, caption="Life-Style Advices Powered by OpenAI API", use_column_width=False, width=100)
with col3:
    st.image(chatgpt_logo, caption="Health APPs by ChatGPT", use_column_width=False, width=100)
with col4:
    st.image(chatbot_logo, caption="User Questions Answered by ChatBOT", use_column_width=False, width=100)

# Functions to preprocess user input
def preprocess_input(user_input):
    # Combining user input into a DataFrame
    user_df = pd.DataFrame({
        'age': [user_input[0]],
        'sex': [sex_mapping[user_input[1]]],
        'chestpaintype': [cp_mapping[user_input[2]]],
        'restingbp': [user_input[3]],
        'cholesterol': [user_input[4]],
        'fastingbs': [bs_mapping[user_input[5]]],
        'restingecg': [ecg_mapping[user_input[6]]],
        'maxhr': [user_input[7]],
        'exerciseangina': [angina_mapping[user_input[8]]],
        'oldpeak': [user_input[9]],
        'st_slope': [slope_mapping[user_input[10]]]
    })

    return user_df

# Function to train the model with hyperparameter tuning
def train_model():
    # Features and target variable
    X = heart_data.drop('heartdisease', axis=1)
    y = heart_data['heartdisease']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Increasing the number of trees
    model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1)

    # Training the model
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Calculating the model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    return model

# Function to create scatter plot for user input
def plot_user_input(user_input):
    user_input_labels = ['age', 'sex', 'chestpaintype', 'restingbp', 'cholesterol', 'fastingbs', 'restingecg', 'maxhr', 'exerciseangina', 'oldpeak', 'st_slope']

    # Excluding non-numeric values
    user_input_values = [float(value) if isinstance(value, (int, float)) else None for value in user_input]

    # Removing None values for plotting
    user_input_labels = [label for label, value in zip(user_input_labels, user_input_values) if value is not None]
    user_input_values = [value for value in user_input_values if value is not None]

    if user_input_values:
        fig, ax = plt.subplots(figsize=(4, 3))
        # Scatter plot for user input
        ax.scatter(user_input_labels, user_input_values, color='orange', marker='o', s=3*25) 
        ax.set_ylabel('Values')
        ax.set_title('User Input Values')
        st.pyplot(fig)
    else:
        st.warning("No numeric values to display.")

# Function to generate advice using OpenAI API
def generate_advice(user_input):
    prompt = f"Given the user input: {user_input}, provide advice to reduce the risk of heart disease."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1
    )

    advice = response['choices'][0]['message']['content']
    return advice

# Function to generate ChatGPT response
def generate_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1
    )

    return response['choices'][0]['message']['content']

# Function to create chat box and interact with ChatGPT
def chat_with_gpt():
    st.subheader("Chat with Hearty, Your AI Support Companion:")
    st.write("Hello! My name is Hearty, your dedicated support companion! Celebrate your journey towards transformative lifestyle changes and remarkable achievements. How may I assist you on this empowering path? I can provide personalized advice tailored just for you.")
    user_message = st.text_input("You:", "")
    if user_message:
        response = generate_chatgpt_response(user_message)
        st.text_area("ChatGPT:", value=response, height=200, max_chars=None, key=None)

# Creating the main function
def main():
    # Adding the CSS styles
    with open("static/style.css", "r") as f:
        custom_css = f.read()
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)
    
    # Rendering the HTML file from the static folder
    st.markdown(open("static/index.html").read(), unsafe_allow_html=True)

    # Setting the title of the Streamlit app
    st.title("Random Forest Classifier Model Prediction")

    # Training the model
    model = train_model()

    # Creating the sidebar for user input
    st.sidebar.header("User Input Features")

    # Setting numeric features
    age = st.sidebar.slider("Age", int(heart_data['age'].min()), int(heart_data['age'].max()), int(heart_data['age'].median()))
    resting_bp_input = st.sidebar.slider("Resting Blood Pressure", int(heart_data['restingbp'].min()), int(heart_data['restingbp'].max()), int(heart_data['restingbp'].median()))
    cholesterol_input = st.sidebar.slider("Cholesterol", int(heart_data['cholesterol'].min()), int(heart_data['cholesterol'].max()), int(heart_data['cholesterol'].median()))
    max_hr_input = st.sidebar.slider("Max Heart Rate", int(heart_data['maxhr'].min()), int(heart_data['maxhr'].max()), int(heart_data['maxhr'].median()))
    oldpeak_input = st.sidebar.slider("Oldpeak", float(heart_data['oldpeak'].min()), float(heart_data['oldpeak'].max()), float(heart_data['oldpeak'].median()))

    # Setting categorical features
    sex_input = st.sidebar.selectbox("Sex", ['M', 'F'])
    cp_input = st.sidebar.selectbox("Chest Pain Type", ['NAP', 'ATA', 'ASY', 'TA'])
    bs_input = st.sidebar.selectbox("Fasting Blood Sugar", ['Normal', 'ST'])
    ecg_input = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST'])
    angina_input = st.sidebar.selectbox("Exercise-Induced Angina", ['N', 'Y'])
    slope_input = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

    # Combining user input into a list
    user_input = [age, sex_input, cp_input, resting_bp_input, cholesterol_input, bs_input, ecg_input, max_hr_input, angina_input, oldpeak_input, slope_input]

    # Displaying user input
    st.subheader("User Input:")
    st.write(f"Age: {age}, Sex: {sex_input}, Chest Pain Type: {cp_input}, Resting BP: {resting_bp_input}, Cholesterol: {cholesterol_input}, Fasting BS: {bs_input}, Resting ECG: {ecg_input}, Max HR: {max_hr_input}, Exercise Angina: {angina_input}, Oldpeak: {oldpeak_input}, ST Slope: {slope_input}")

    # Preprocessing user input
    user_df = preprocess_input(user_input)

    # Making predictions on user input
    prediction = model.predict(user_df)

    # Displaying prediction
    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.write("The predicted outcome is: No Risk of Heart Disease")
    else:
        st.write("The predicted outcome is: Risk of Heart Disease")

    # Generating and displaying lifestyle advice using OpenAI API
    st.subheader("To reduce the risk of heart disease, it's important to maintain a healthy lifestyle. You can start by following these suggested tips:")
    advice = generate_advice(user_input)
    st.write(advice)

    # Adding OpenAI logo
    openAI_logo = "Resources/OpenAI.png"
    st.image(openAI_logo, caption="Powered by OpenAI", use_column_width=False, width=150)

    # Generating and displaying ChatGPT response
    chatgpt_prompt = "Share best apps that will help me stay healthy, count my walking distance. Start your sentence with: 'Here is a list of apps that will help you stay healthy:"
    chatgpt_response = generate_chatgpt_response(chatgpt_prompt)
    st.subheader("ChatGPT Response:")
    st.write(chatgpt_response)

    # Adding ChatGPT logo
    chatgpt_logo = "Resources/ChatGPT.png"
    st.image(chatgpt_logo, caption="Produced by ChatGPT", use_column_width=False, width=150) 

    # Creating a scatter plot for user input
    plot_user_input(user_input)

    # Adding a chat with ChatGPT for user interaction
    chat_with_gpt()

    # Adding ChatBOT `Hearty` Character logo
    hearty_logo = "Resources/Hearty.png"
    st.image(hearty_logo, caption="Answered by ChatBOT", use_column_width=False, width=75)

    # Rendering the footer
    st.markdown("""
    <footer class="footer">
        <p>2024 CardioGuard AI. All rights reserved.</p>
        <p>Many thanks to our TA`s: Tom-Jordan-Yash-Mohammed and Sureer<p>  
        <p>for their effort and great support during our data journey!</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# CTRL+C to stop app 
#------------------------------------------------------------------------------------------------------------
# NOTES

# right-click app6.py and open in integrated terminal
# activate venv (virtual environment) by running: source ./venv/Scripts/activate 
# dont forget to install all below in venv (virtual environment)
# OR copy below code into terminal (one line at a time)
# python -m venv venv
# source ./venv/Scripts/activate

# if running the app gives errors first try: streamlit cache clear
# and also clear terminal by clicking 3 dots on top right of terminal and click clear terminal 
# clear your browser cache
# change Resources3 folder name, Heart3.csv file name, and app3.py file name

# if problem continues run below in integrated terminal one line at a time
# deactivate
# rm -rf venv  # Use 'rmdir /s /q venv' on Windows
# python -m venv venv
# on Windows : .\venv\Scripts\activate
# on Mac : source venv/bin/activate
    
# if problem continues run: python.exe -m pip install --upgrade pip
# pip install streamlit==1.29.0
# pip show streamlit

# run the application by running: python app.py
# OR copy below code into terminal (one line at a time)
# flask shell
# from app import db
# db.create_all()
# exit()
# flask run

# -----------------------------------------------------------------------------------------------------------------------------
# Using a Dynamic Update (No Submit Button):
# Real-time Feedback: Users get immediate feedback as they make selections, creating a more dynamic and responsive user interface.
# Streamlined Interaction: Reduces the number of explicit steps, making the interface feel more seamless and modern.
# Automatic Submission: The form is automatically submitted when any selection changes, potentially reducing the need for an explicit submit action.
# Considering working with a machine learning model, the choice might depend on factors such as the processing time for the predictions and whether we want to provide real-time feedback as users make selections.
# If the processing time for predictions is minimal, providing real-time feedback without a submit button can enhance the user experience. However, if predictions involve heavy computation or server-side processing, having a submit button allows us to control when the predictions are triggered, providing a smoother experience for the user.

# ------------------------------------------------------------------------------------------------------------------------------
# STREAMLIT APP 
# Below Streamlit app takes user input for various features related to heart health and predicts whether the user is likely to have heart disease or not. 
# The model used for prediction is a RandomForestClassifier. The app also provides model evaluation metrics, including accuracy, 
# confusion matrix, and classification report based on a test

# activate venv (virtual environment): source ./venv/Scripts/activate
# then run: streamlit run app.py
# dont forget to CTRL+C to stop app    
