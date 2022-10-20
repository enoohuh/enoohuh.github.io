# Enoque's Data Science Portfolio

# Can we predict how good a wine tastes?
* Utilized different regression models to predict wine quality ratings from a list of objective features of the wine (pH level, residual sugar, alcohol content, etc.)
* Compared the performance of 4 different models: ordinary least squares, ridge regression, lasso regression, and elastic net.
* Utilized cross validation to choose the tuning parameters in order to improve model performance.
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

