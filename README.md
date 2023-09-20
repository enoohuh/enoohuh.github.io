# Enoque's Data Science Portfolio
## Topics Overview
1. [Regression](#wine)
2. [Machine Learning and Predictive Modeling](#election)
3. [Data Visualization](#transportation)
# What Exactly Makes Wine Taste Good? <a name="wine"></a>
### Introduction
* Wine preferences are very subjective as people tend to like different things. But is there a way to create an objectively good tasting wine with the help of machine learning? In this project, we aimed to find out what exactly makes wine taste good, as we analyzed the similarities and differences of 1600 wines. Objective features of the wine included things like: fixed acidity, residual sugars, pH level, alcohol content, etc. Each wine was rated by three different wine experts and the median of the ratings was used as the wine score which ranged from 0 (very bad) to 10 (excellent).
### Methods
* Four different regression models were used to compare results: ordinary least squares, ridge regression, lasso regression, and elastic net. Cross-validation was used for choosing the tuning parameters of all models except ordinary least squares. All code was written in Python, and packages used were: numpy and pandas for data manipulation, seaborn for data visualization, and scikit-learn for model training and cross-validation.
### Model Creation
#### Ordinary Least Squares
* We want to identify the coefficients of a linear model relating wine quality to different features of the wine. Our predictors are all of the features of the wine, and our response variable is the subjective rating that each wine was given by the wine experts. The complete list of 11 features includes: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, and alcohol. The model was split 70/30 as train/test data.
#### Ridge Regression
* To optimize our ridge regression model, we utilized the default leave-one-out cross validation, and inputted a list of $\alpha$ values $[0.1, 0.11, 0.12, ..., 2]$ where higher values of $\alpha$ correspond to stronger regularization. The final $\alpha$ value was 0.21 as shown by the graph below. We see that when $\alpha=0.21$, the mean squared error is the lowest, and as $\alpha$ increases past 0.21, the mean squared error increases rapidly. This shows that some regularization can help model performance, while too much regularization can reduce model performance.
![](/images/ridge.png)
#### Lasso Regression
* For the lasso model, 5-fold cross-validation was used. We inputted a list of $\alpha$ values $[0.001, 0.002, 0.003, ..., 1]$ and the amount of penalization chosen was $\alpha=0.001$. We observe that in contrast to ridge regression, lasso regression gets rid of some features completely, as residual sugar as well as density are now both 0. This is because lasso (L1 regularization) is considered a more strict shrinkage operation, and leads to sparser models.
![](/images/lasso.png)
#### Elastic Net
* In an elastic net model, there are 2 tuning parameters we need to consider when using cross-validation: the L1 ratio and $\alpha$. 
L1 ratio = 0 is ridge regression
L1 ratio = 1 is lasso regression.
Our model's L1 ratio chosen by cross-validation was 1, so in this case our elastic net model is the same as a lasso regression model.
### Results
![](/images/wine%201.jpg)
![](/images/Wine%20image%202.jpg)

# Can We Use Machine Learning to Predict Primary Election Outcomes? <a name="election"></a>
### Introduction
* In this study, I explored the question of: "Can we use data to predict the results of a primary election?" We aimed to create a more complete representation of each candidate by combining two main data sets: FiveThirtyEight's Election Candidates data set and the Federal Election Commision's (FEC) campaign finance data set. By doing so, we were able to take advantage of many useful features like their demographics, finances, and endorsements. All code was written in Python and packages used included: numpy and pandas for data manipulation, seaborn and matplotlib for data visualization and  scitkit-learn for machine learning.
### Data Cleaning
* FiveThirtyEight's election candidates data set was merged with FEC's campaign finance data set. Each data set had a distinct naming convention, so we dropped middle names, suffixes, and nicknames and instead matched based on first name and last name. Certain features like date of transaction and candidates' identification numbers were dropped from the dataframe, as they did not provide any relevant and quantifiable information.
### Feature Engineering
* The Federal Election Commission data set contained some double-counted finance data. Some columns such as total receipts, total disbursements and candidate contributions were either combined or subtracted from each other to prevent double counting and collinearity of features. In addition, some binary endorsement data from the other data set such as "Endorsed by Trump?, "Endorsed by Sanders?", and "Obama Alum?" were one-hot-encoded in order to make model creation more efficient.
### Visualizing Trends
* How much would being endorsed by big names like Trump, Biden, Sanders, and Warren help candidates win a primary election? Do these big names often endorse or anti-endorse a lot of people? We also wanted to visualize the financial support that candidates received for their campaigns. How many individual contributions do candidates usually receive?
### Machine Learning
#### Decision Trees and Random Forests
* We utilized scikit-learn's decision tree classifier model as well as the random forest classifier model. We utilized a 70/30 split for train/test data. Because of the huge number of features from the combined data set, we changed the parameter "max depth" to set a limit to the number of nodes and prevent overfitting on our test data.
#### Limitations of Our Models
* Although most features included in both Republican and Democratic data sets were the same, the Democratic data set had a handful of extra features that the Republican data set did not have. It included features such as: whether the candidate was white, a veteran, an elected official, a self-funder, had a STEM background, etc. These differences in features between the data sets might have helped or not helped the accuracy of our model. Although being a self-funder would probably mean the candidate is less known and has less support, we can't exactly quantify the impact of these features nor can we attribute the differences in results to these factors.
### Results
* Our models were surprisingly accurate in predicting who will win a primary election. Both the random forest model and the decision tree model performed similarly in accuracy on test data. In order to get a more complete summary of our models' performances, we also looked at precision and recall which gives us a better understanding of the true positives in our predictions. 
![](/images/endorsement%20image.jpg)
![](/images/Election%20proj%20image.jpg)


# The impact of COVID-19 on New York City's Public Transportation System <a name="transportation"></a>
### Introduction
* This project aimed to summarize and create a comprehensive report on the impact of COVID-19 on New York City's public transportation system. We focused on two main public transportation systems that millions of New Yorkers rely on daily: the subway and the bus. Some questions we explored were: "How has COVID-19 affected public transportation ridership?" and "What are factors that correlate with ridership numbers slowly going back to normal?"
### Methodology
* All the code was written in Python. Main packages include: numpy and pandas (for data cleaning and manipulation), matplotlib and seaborn (for data visualization), folium (for the interactive map), and scikit-learn (for machine learning). 
Data were extracted mainly from the Metropolitan Transportation Authority website, and New York State Department of Transportation's website.
### Data Visualization
#### Subway and Bus Riderships
* Of all New Yorkers who commute, 39 percent use the subway, and 11 percent take the bus. With the issue of an executive order to close non-essential businesses in March of 2020, public transportation services in New York City remained open, but were reduced significantly. We can see the dip in both subway and bus ridership in the figures below. We can also see the gradual rise in ridership in the following months, as safety measures like masks, sanitation, and social distancing were introduced.
#### Correlation with COVID-19 Cases
* We also explored the potential correlation between public transportation ridership and COVID-19 cases. We wanted to see if there was an inverse correlation between ridership numbers and COVID-19 cases. For example, if people get sick with COVID-19, do they still go out and use public transportation? Or do they stay home and follow the stay-at-home mandates that were in place? We also wanted to explore if the sentiment remained constant from 2020 to 2021. Did people take the pandemic more seriously in the beginning? We can see this phenomenon, as in early 2020, a rise in COVID-19 cases correlated with a stark drop in ridership numbers. In contrast, a rise in COVID-19 cases in early 2021 didn't seem to have much impact on ridership numbers at all, as they're seen steadily increasing back to normal pre-pandemic levels.
![](/images/nyc%20page%201.jpg)
![](/images/nyc%20page%202.jpg)
![](/images/nyc%20page%203.jpg)

