# CardioGuard-AI
Predicting Heart Disease Risk - with Machine Learning Models - Providing Lifestyle Advice using AI


<img src="https://github.com/YargKlnc/CardioGuard-AI/assets/142269763/a768f388-2491-4adf-b79e-72deead8168a" alt="Image" width="50%" height="50%">

![image](https://github.com/YargKlnc/CardioGuard-AI/assets/142269763/9d78b518-6d10-472f-9fec-4cade7011953)

![image](https://github.com/YargKlnc/CardioGuard-AI/assets/142269763/f60010cd-4b61-4b8a-a3aa-9be6c2cc7d2c)

![image](https://github.com/YargKlnc/CardioGuard-AI/assets/142269763/a2fa7bf4-bc77-4950-8126-ebf1a9360036)

![image](https://github.com/YargKlnc/CardioGuard-AI/assets/142269763/5aec70de-d2e6-45f0-9ae2-ac4b47bfc37a)


This innovative project leverages advanced technologies to create an interactive and insightful Heart Disease Prediction App. Utilizing the power of OpenAI's GPT-3.5-turbo model, the application seamlessly integrates machine learning, natural language processing, and data visualization to empower users in understanding and predicting their risk of heart disease. By importing and analyzing a PostgreSQL dataset containing crucial health parameters, the system employs a RandomForestClassifier for accurate predictions. The user-friendly interface, developed using Streamlit, enables individuals to input both numeric and categorical health features, generating predictions and personalized lifestyle advice. This comprehensive tool not only offers predictive analytics but also engages users in a meaningful way by providing health advice through state-of-the-art natural language processing. The project stands at the intersection of cutting-edge technology and healthcare, showcasing the potential for AI to enhance health-related decision-making and promote proactive well-being.

Moreover, the application integrates OpenAI's ChatGPT to provide users with tailored lifestyle advice, adding a unique dimension to the user experience by combining expert predictions with personalized guidance from advanced natural language processing. This collaborative effort between OpenAI and the predictive modeling capabilities of the Heart Disease Prediction App underscores the potential for AI to revolutionize health awareness and decision-making.

Extended Key Features:

1. **Cloud PostgreSQL Integration:** Utilizes ElephantSQL to connect to a PostgreSQL database, allowing seamless access to heart health data.
2. **CSS Styling:** Incorporates CSS styling from an external file to enhance the visual appeal and overall aesthetics of the application.
3. **Background Customization:** Sets a light orange background color, creating a visually soothing and cohesive theme for the app.
4. **Categorical Mapping:** Maps categorical features such as sex, chest pain type, and more to numeric values for effective model training.
5. **Numeric Feature Visualization:** Displays scatter and bar charts showcasing numeric features, providing users with valuable insights.
6. **Responsive Design:** Ensures a responsive design that adapts to various screen sizes, enhancing accessibility for users on different devices.
7. **Hyperparameter Tuning:** Optimizes model performance through hyperparameter tuning, fine-tuning parameters for increased accuracy.
8. **Dynamic Plotting:** Dynamically updates scatter and bar charts based on user input, offering real-time visualizations for a more interactive experience.
9. **Lifestyle Advice Generation:** Utilizes OpenAI's ChatGPT to generate personalized lifestyle advice, enhancing user engagement and health awareness.
10. **Interactive Sidebar:** Implements a sidebar for user input features, providing a streamlined and intuitive interface for data entry.
11. **Model Training:** Trains the RandomForestClassifier on heart health data, ensuring the model is well-equipped to make accurate predictions.
12. **Predictive Modeling:** Demonstrates the power of predictive modeling in healthcare, showcasing its potential for early risk detection.
13. **User-Centric Approach:** Focuses on user experience by combining predictive analytics with natural language processing for a holistic health assessment.
14. **OpenAI Collaboration:** Integrates OpenAI's ChatGPT for advanced language processing, offering users not only predictions but also valuable insights and lifestyle advice.

This comprehensive set of features collectively establishes the Heart Disease Prediction App as a cutting-edge and user-centric tool for health risk assessment.

Summary of the Code:

1. **Imported required libraries:** `openai`, `streamlit`, `pandas`, `sqlalchemy`, `RandomForestClassifier`, `train_test_split`, `accuracy_score`, and `matplotlib.pyplot`.
2. **Set OpenAI API key** and created a PostgreSQL database engine.
3. Loaded heart disease **data from PostgreSQL** and renamed columns to match PostgreSQL names.
4. **Loaded CSS styles for customization**.
5. **Defined mappings for categorical features** like sex, chest pain type, fasting blood sugar, etc.
6. **Converted categorical columns to numeric** in the original dataset using the defined **mappings**.
7. **Displayed an image of a healthy heart** in the header.
8. **Defined a function to preprocess user input** into a DataFrame.
9. Defined a function to train a **RandomForestClassifier model** with hyperparameter tuning.
10. Created **a sidebar in Streamlit** for user input features (numeric and categorical).
11. Displayed user input and preprocessed it **for model prediction**.
12. Made predictions on user input using the trained model and displayed **the predicted outcome**.
13. Plotted **a scatter plot for user input** values and a bar chart for mean values of numeric features in the dataset.
14. Generated **lifestyle advice using the OpenAI GPT-3.5-turbo** model based on user input.
15. Displayed **ChatGPT response** and logos for both OpenAI and ChatGPT.


**Data cleaning, modelling and preprocessing**

The source dataset was very clean that required minimal cleaning. We ensured this by conducting basic data exploration. We used .describe to see key statistics of the data and also checked and dropped any NaN values.  We have a good split between the records that are heart disease and the records that are not heart disease. We have 508 records for heart disease and 430 records for non heart disease.  To enhance the model’s understanding, in our preprocessing we mapped the categorical features so that our model can make accurate predictions. 

Data cleaning, modelling, and preprocessing can be found in the file Heart_Prediction_Elected_RF_Model.ipynb
**Machine Learning Optimization**

Machine Learning Model chosen: RandomForest Classifier 

With our app being predicting heart disease and having features that either indicated a patient had heart disease or did not, we knew we wanted to test classifier models. We first tried the RandomForestClassifer, this was our favoured model going into this project as our research indicated this model is used in other apps similar to ours. We were able to achieve a 0.86 accuracy score after three optimizations. To ensure we were not being biassed towards RandomForest Classifer we also tested Gradient Boosting Classifer and SVM which got us an accuracy score of 0.83 and 0.82 respectively.

We had three different optimization trials with the RandomForest Classifier. The first two trials were completed after the data had been preprocessed with categorical mapping and scaled using standard scaler. We were only able to reach an accuracy score of 0.82, while this was still good, we wanted to be better. Our research gave us the idea to try not scaling the data, we kept the categorical mapping preprocessing but by removing the scaling, we were able to get an accuracy score of 0.86 which we were very happy with.

Note: Ensure to replace the OpenAI API key with your actual key, copying to your virtual environment or adding to the environment variables in your local machine before running the code.


**References**

Heart Failure Prediction Dataset by: www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

AI Functions and API Key: https://openai.com/

Head Photo Rights: https://www.bhf.org.uk/what-we-do/our-research/research-successes/artificial-intelligence-and-heart-attack
