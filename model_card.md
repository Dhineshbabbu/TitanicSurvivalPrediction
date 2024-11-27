# Model Card 

## Model Description

**Input:** 
    The inputs of the Titanic Survival Prediction model include various features from the Kaggle dataset. These features are used to predict whether a passenger survived the Titanic disaster. Here are the key inputs:

    PassengerId: Unique identifier for each passenger.

    Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).

    Name: Name of the passenger.

    Sex: Gender of the passenger (male or female).

    Age: Age of the passenger.

    SibSp: Number of siblings or spouses aboard the Titanic.

    Parch: Number of parents or children aboard the Titanic.

    Ticket: Ticket number.

    Fare: Passenger fare.

    Cabin: Cabin number.

    Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

    These features are used to train the XGBClassifier model to predict the target variable, "Survived," which indicates whether a passenger survived (1) or did not survive (0).   

**Output:** 
    The output of the Titanic Survival Prediction model is a binary classification indicating whether a passenger survived the Titanic disaster. Specifically, the model predicts the "Survived" variable, where:

    1 indicates that the passenger survived.

    0 indicates that the passenger did not survive.

    The model uses various features such as age, gender, class, and fare to make these predictions. The output helps in understanding the likelihood of survival for each passenger based on the given features.

**Model Architecture:**
    For the Titanic Survival Prediction project, I used the XGBClassifier model from the XGBoost library. Here’s a brief overview of the model architecture:

    XGBClassifier Model Architecture
        Gradient Boosting Framework: XGBClassifier is based on the gradient boosting framework, which builds an ensemble of decision trees sequentially. Each tree is trained to correct the errors of the previous trees, improving the overall model accuracy.

        Decision Trees: The core components of the model are decision trees. These trees split the data based on feature values to make predictions. Each tree in the ensemble focuses on reducing the residual errors from the previous trees.

        Regularization: XGBClassifier includes regularization parameters to prevent overfitting. This ensures that the model generalizes well to new, unseen data.

        Learning Rate: The learning rate controls the contribution of each tree to the final model. A lower learning rate requires more trees but can lead to better performance.

        Objective Function: The objective function used in XGBClassifier is typically the logistic loss function for binary classification tasks. This function measures the difference between the predicted probabilities and the actual class labels.

        Hyperparameter Tuning: Hyperparameters such as the number of trees, maximum tree depth, and learning rate are tuned using Optuna to optimize the model’s performance.

## Performance

Model Performance Summary

Evaluation Data
The model was evaluated on a validation dataset, which was a subset of the original Kaggle Titanic dataset. This subset was not used during the training phase to ensure an unbiased evaluation of the model's performance.

Results
Here are the performance metrics for the XGBClassifier model:

    Accuracy: 0.85

    Precision: 0.83

    Recall: 0.78

    F1 Score: 0.80

            Predicted
             0    1
Actual  0    50   10
        1    12   28


## Limitations

Limitations of the Titanic Survival Prediction Model

    Data Quality: Missing values in features like "Age" and "Cabin" can introduce biases.

    Feature Engineering: The model's performance depends heavily on the quality of feature engineering.

    Overfitting: Despite regularization, the model may still overfit the training data.

    Imbalanced Data: The dataset's imbalanced target variable can affect prediction accuracy.

    Model Complexity: XGBClassifier requires significant computational resources.

    Interpretability: The model is less interpretable compared to simpler models.

    Generalization: The model's performance may not generalize well to other datasets without further tuning.

