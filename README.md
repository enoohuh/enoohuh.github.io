# Enoque's Data Science Portfolio

# Can we predict how good a wine tastes?
### Introduction
* Wine preferences are very subjective as people tend to like different things. But is there a way to create an objectively good tasting wine with the help of machine learning? In this project, we aimed to find out what exactly makes wine taste good, as we analyzed the similarities and differences of 1600 wines. Objective features of the wine included things like: fixed acidity, residual sugars, pH level, alcohol content, etc. Each wine was rated by three different wine experts and the median of the ratings was used as the wine score which ranged from 0 (very bad) to 10 (excellent).
### Methods
* Four different regression models were used to compare results: ordinary least squares, ridge regression, lasso regression, and elastic net. Cross-validation was used for choosing the tuning parameters of all models except ordinary least squares. All code was written in Python, and packages used were: numpy and pandas for data manipulation, seaborn for data visualization, and scikit-learn for model training and cross-validation.
### Model Creation
#### Ordinary Least Squares
* We want to identify the coefficients of a linear model relating wine quality to different features of the wine. Our predictors are all of the features of the wine, and our response variable is the subjective rating that each wine was given by the wine experts. The complete list of 11 features includes: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, and alcohol. The model was split 70/30 as train/test data.
#### Ridge Regression
* To optimize our ridge regression model, we utilized the default leave-one-out cross validation, and inputted a list of $\alpha$ values $[0.1, 0.11, 0.12, ..., 2]$ where higher values of $\alpha$ correspond to stronger regularization. The final $\alpha$ value was 0.21 as shown by the graph below. We see that when $\alpha=0.21$, the mean squared error is the lowest, and as $\alpha$ increases past 0.21, the mean squared error increases rapidly. This shows that some regularization can help model performance, while too much regularization can reduce model performance.
#### Lasso Regression
* For the lasso model, 5-fold cross-validation was used. We inputted a list of $\alpha$ values $[0.001, 0.002, 0.003, ..., 1]$ and the amount of penalization chosen was $\alpha=0.001$. We observe that in contrast to ridge regression, lasso regression gets rid of some features completely, as residual sugar as well as density are now both 0. This is because lasso (L1 regularization) is considered a more strict shrinkage operation, and leads to sparser models.
#### Elastic Net
* In an elastic net model, there are 2 tuning parameters we need to consider when using cross-validation: the L1 ratio and $\alpha$. 
L1 ratio = 0 is ridge regression
L1 ratio = 1 is lasso regression.
Our model's L1 ratio chosen by cross-validation was 1, so in this case our elastic net model is the same as a lasso regression model.
### Results
![](/images/wine%201.jpg)
![](/images/Wine%20image%202.jpg)

# Estimating the _causal effect_ of a childcare treatment on the cognitive test scores of premature children
* The Infant Health and Development Program (IHDP) was an experiment treating low-birth-weight, premature infants with intensive high-quality childcare from a trained provider. I estimated the causal effect of this program on the children's cognitive test scores.
* Devised a propensity score model to control for confounders (child's birth weight, whether or not the mother smoked during pregnancy, her age, race, etc.)
* Calculated and compared the estimate of the average treatment effect (ATE) and the simple difference in mean outcomes (SDO)
![](/images/causal%20inf%20image.jpg)

# Can we use campaign data to predict who wins a primary election?
* Combined two datasets to create a more complete representation of each candidate. The first dataset was FiveThirtyEight's 2018 Primary Election dataset which contained endorsement data. The second dataset was the Federal Election Commission's dataset which contained candidates' finance data (individual contributions to the campaign, total disbursements, etc.)
* Utilized a random forest classifier model to predict who wins a primary election based on these combined features.
![](/images/endorsement%20image.jpg)
![](/images/Election%20proj%20image.jpg)

# Trump Tweets
* Created data visualizations from Donald Trump's tweets.
* During the 2016 presidential campaign, it was theorized that Donald Trump's tweets from Android devices were written by him personally, and the tweets from iPhones were from his staff. I plotted the number of tweets from each device and the time when they were usually posted to see if our data supported this claim.
* Utilized VADER, a sentiment analysis algorithm to observe the difference of polarity scores of certain words used in Trump's tweets like "nytimes" and "fox".
![](/images/github%20trump%20tweets%20image.jpg)

# Classifying which emails are spam
* Utilized logistic regression to classify emails in order to predict whether they are spam emails or real emails.
* Feature engineered a list of the most common words used in both spam emails and real emails.
* The model obtained a 93.5% accuracy on test data.
![](/images/Spam%20Ham%20proj%20image.jpg)

# How has the pandemic affected New York City's public transportation ridership?
* Compared two main public transportations: subway and bus. How many people relied on these public transportations before the pandemic? How many of them still use public transportation during the pandemic?
* Plotted the number of COVID-19 cases against public transportation ridership. Does a spike in COVID-19 cases directly correlate with less people riding public transportation?
![](/images/nyc%20page%201.jpg)
![](/images/nyc%20page%202.jpg)
![](/images/nyc%20page%203.jpg)

