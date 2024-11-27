# Titanic Survival Prediction


## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
The Titanic Survival Prediction project aims to predict the survival of passengers on the Titanic using machine learning models. The project involves data preprocessing, feature engineering, and model building. Various algorithms, such as Support Vector Machine (SVM), Neural Network, and Random Forest, are used to create predictive models. The project also includes hyperparameter tuning to optimize model performance. The XGB Classifier model is employed to enhance prediction accuracy, leveraging its ability to handle complex data patterns and interactions. This project provides insights into the factors influencing survival and demonstrates the application of machine learning techniques to historical data.


## DATA
The Titanic Survival Prediction project uses a dataset from Kaggle, where the target variable is "Survived." This variable indicates whether a passenger survived (1) or did not survive (0) the Titanic disaster. The dataset includes various features such as passenger age, gender, class, fare, and more, which are used to build predictive models. The goal is to analyze these features and predict the likelihood of survival for each passenger using machine learning techniques.


## MODEL 
The Titanic Survival Prediction project uses the XGBClassifier from the XGBoost library to predict passenger survival. XGBClassifier is a powerful algorithm known for its efficiency and accuracy in classification tasks. It builds an ensemble of decision trees, corrects errors from previous trees, and includes regularization to prevent overfitting. This model is particularly effective for handling complex data patterns and interactions, making it a suitable choice for this project. The dataset from Kaggle includes various features like age, gender, and class, with "Survived" as the target variable.


## HYPERPARAMETER OPTIMSATION
In the Titanic Survival Prediction project, I used hyperparameter tuning with Optuna to optimize the performance of the XGBClassifier model. Optuna is an efficient and flexible hyperparameter optimization framework that uses a trial-based approach to find the best hyperparameters. By leveraging Optuna, I was able to systematically explore the hyperparameter space and identify the optimal settings for the model, resulting in improved accuracy and robustness. This approach helped enhance the predictive power of the model, making it more effective in predicting passenger survival.


## RESULTS
n the Titanic Survival Prediction project, I used the XGBClassifier model to predict passenger survival. The dataset from Kaggle included features like age, gender, and class, with "Survived" as the target variable. I applied hyperparameter tuning using Optuna to optimize the model's performance. The tuned model achieved high accuracy, highlighting key factors influencing survival. This project demonstrated the effectiveness of advanced machine learning techniques and hyperparameter optimization in building robust predictive models.


