# Titanic-Data-ML-Spark

The Titanic Data ML Spark problem involves predicting the survival of passengers on the Titanic based on various features such as age, gender, passenger class, etc. The goal is to build a machine learning model using Spark and predict whether a passenger survived or not.

# Dataset Description

1.PassengerId: An integer that uniquely identifies each passenger.

2.Survived: An integer indicating whether the passenger survived (1) or not (0).

3.Pclass: An integer representing the passenger class (1, 2, or 3) indicating the socio-economic status of the passenger.

4.Name: A string representing the name of the passenger.

5.Sex: A string indicating the gender of the passenger (male or female).

6.Age: A double representing the age of the passenger.

7.SibSp: An integer representing the number of siblings/spouses the passenger had aboard the Titanic.

8.Parch: An integer representing the number of parents/children the passenger had aboard the Titanic.

9.Ticket: A string representing the ticket number.

10.Fare: A double representing the fare (price) the passenger paid for the ticket.

11.Cabin: A string representing the cabin number.

12.Embarked: A string indicating the port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

# Model Building

* Importing the necessary libraries: The code starts by importing the required libraries, including pyspark, SparkSession, and various classes from pyspark.ml and pyspark.sql modules.

* Creating a SparkSession: The code creates a SparkSession, which is the entry point for interacting with Spark.

* Loading the Titanic dataset: The code reads the Titanic dataset from a CSV file using spark.read.csv(). The dataset is assigned to the titanic_df DataFrame.

* Exploratory data analysis (EDA): The code performs various exploratory data analysis tasks to understand the dataset. It includes printing the row count, displaying the first 5 rows of the DataFrame, describing the dataset, printing the schema, selecting and displaying specific columns, and performing groupBy operations to analyze survival based on different attributes.

* Handling missing values: The code defines a function null_value_count() to identify columns with null values and their counts. It is then used to print the columns with null values in the Titanic dataset. The code also replaces missing values in the "Age" column based on the average age of different salutations in the "Name" column.

* Data preprocessing: The code performs several data preprocessing steps. It fills the missing values in the "Embarked" column, drops the "Cabin" column due to many null values, creates new features like "Family_Size" and "Alone" based on existing columns, converts categorical columns ("Sex", "Embarked", "Initial") into numerical representations using StringIndexer, and drops unnecessary columns.

* Feature vectorization: The code uses VectorAssembler to combine all the features into a single vector column named "features" in the DataFrame.

* Splitting the dataset: The code splits the dataset into training and testing datasets using randomSplit(). The training dataset contains 80% of the data, and the testing dataset contains 20%.

* Model training and evaluation: The code trains multiple classification models, including Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient-boosted Tree Classifier, Naive Bayes, and Support Vector Machine. It uses the training dataset to train the models and applies them to the testing dataset to make predictions. The accuracy of each model is evaluated using the MulticlassClassificationEvaluator and printed.
