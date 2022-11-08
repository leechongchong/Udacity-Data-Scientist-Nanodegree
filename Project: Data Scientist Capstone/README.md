# Credit Card Fraud Detection using Machine Learning Algorithms
## Project Motivation
The risk of fraudulent credit card has become one of the most serious problems plaguing global banks. Taking the United States as an example, the Federal Reserve's Payment Investigation Report shows that in 2012, the total amount of credit card payments in the United States reached 26 billion US dollars, of which unauthorized credit card payments, that is, the amount of fraudulent credit card transactions, was as high as 6.1 billion US dollars. Measuring the risk of credit card fraud involves a complex range of techniques, from finance to economics to law to information science. Traditional credit card fraud detection requires a lot of manpower to analyze and judge. Human auditors will usually call to confirm if the transaction is suspected of credit card fraud. Now, due to the surge in transaction volume, the credit card departments of major banks are relying on big data to quickly identify suspected fraudulent credit card transactions through machine learning and cloud computing methods.

This project aims to explore basic applications of machine learning in credit card fraud detection. Computer programming and mathematical models are used to construct the main characteristics of credit card fraud, while machine learning methods are used to automatically identify credit card transactions suspected of fraud. Finally, I generate an intelligent program for real-time monitoring of credit card fraud, which strive to obtain instructive conclusions for practice.

## Files & Libraries Description

### Files
Credit Card Payment Fraud.xlsx \
is a dataset provided by ID Analytics containing 95,007 records of card transaction from 2010-01-01 to 2010-12-31. It includes information about card number, date, merchant’s number, description, state, zip code, transaction type, and amount. Every record is also labeled as fraud or not. In total, there are 298 labeled fraudulent records.
10 variables in total – 1 numeric, 7 categorical, 1 text, 1 date
Numeric: amount
Categorical: recordnum, cardnum, merchnum, merch.state, merch.zip, transtype, and fraud Text: merch.description
Date: date

Credit Card Payment Fraud Clean.xlsx \
is the clean version of Credit Card Payment Fraud.xlsx after the missing values being handled. 

Credit Card Payment Fraud Features.xlsx \
is the final version of Credit Card Payment Fraud.xlsx without newly created features related to fraud detection.

DQR.ipynb \
is the Jupyter Notebook conducting explotary data analysis on the given dataset.

DQR.xlsx \
is the excel workbook with Data Quality Report output from DQR.ipynb

Data Cleaning.ipynb \
is the Jupyter Notebook cleaning the given dataset, especially on tackling missing values. 

Feature Engineering.ipynb \
is the Jupyter Notebook creating new features relevant to detecting credict card payment frauds.

Feature Selection Output.png \
is the visualization of relationship between number of selected features vs. model performance.

Machine Learning Algorithms.ipynb \
is the Jupyter Notebook applying and evaluting different machine learning algorithms on predicting credit card frauds.

Top 10 Features.png \
is the output of Top 10 behaviorial features in detecting fraudulent credict card transactions.


### Libraries
numpy \
offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, and more.

pandas \
is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.

time \
provides various time-related functions.

matplotlib.pyplot \
is a state-based interface to matplotlib. It provides an implicit, MATLAB-like, way of plotting. It also opens figures on your screen, and acts as the figure GUI manager.

sklearn \
is a free software machine learning library for the Python programming language.

xgboost \
belongs to a family of boosting algorithms and uses the gradient boosting (GBM) framework at its core. 


## Results Summary
The technical details, main results and insights of the project can be found at the Medium Blog post available here: https://medium.com/@chongli06/credit-card-fraud-detection-using-machine-learning-algorithms-69cf15f849f8


## Acknowledgement
Thank you Prof.Stephen Coggeshall, Chief Analytics and Science Officer IDA, LifeLock in sharing the data and provided guidance & instructions in general fraud analytics. 

Thank you my classmates Jie Chen, Raman Deep Singh, Xiaowen Zhang, Yu Dong in brainstorming on this project together with me. 



