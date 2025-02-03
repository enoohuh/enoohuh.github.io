# Enoque's Data Science Portfolio
## Topics Overview
* [Project 1.](#llm) Gaming GPT — AI Assistant for Fortnite
* [Project 2.](#wine) Regression ([code](#code2))
* [Project 3.](#election) Machine Learning and Predictive Modeling ([code](#code3))
* [Project 4.](#transportation) Data Visualization ([code](#code4))
* [Project 5.](#fortnite) Sentiment Analysis and Web Scraping ([code](#code5))

# Project 1: A Fine-Tuned GPT Model For Competitive Gaming <a name="llm"></a>
### 1. Introduction
* With the rise of large language models (LLMs) in the recent years, many fields have incorporated AI in order to improve and automate their services. However, there’s one field in particular where LLMs have not been incorporated in any significant way: ***gaming***. This project aims to explore a potentially useful application of LLMs in gaming: an AI assistant that can answers in-game related questions to elevate the player experience.
![](/images/fortnite_llm.jpg)

#### 1.1 What Kind of Data Would Help Players?
* In the world of competitive Fortnite, there are different types of information that are essential to survival in high-pressure situations. The constant changes in the game make it extremely hard for players to memorize all of the necessary information that could potentially be the difference between life and death, or even earning life-changing money from tournaments. Some examples include: weapon information, vending machine locations, "choppas" or helicopter locations, and NPC (non-playable character) locations.
##### Weapon Stats
* With more than 100 weapons currently in the game, memorizing all of this information is not feasible. Stats such as: damage per second (DPS), structural damage, reload time, magazine size, etc. are all valuable pieces of information that are sometimes necessary to know when playing at a competitive level.

![](/images/weapon_stats.jpg)

##### Vending Machine and Choppa Locations
* Vending Machines are spread around the Fortnite map, and they provide players the opportunity to buy heals (medkits, bandages, shield) by using farmed materials (wood, brick, metal) as a form of currency. Knowing where these vending machines spawn is crucial for certain situational scenarios such as: dying in storm, and having no heals but surplus materials.
* Choppas, or helicopters, provide players with extremely fast mobility. It can serve as a tool to rotate to a different part of the map, follow enemies, escape the storm, or disengage from fights.
![](/images/vendingmachine_choppa.jpg)
##### NPC Boss Locations
* NPC Bosses spawn at certain points of interest, and drop "mythic" items once they are defeated. Mythic items are one-of-a-kind weapons or utilities that spawn only once every match. Obtaining these items can be difficult, as many players compete for them. Obtaining these items grant a significant competitive advantage over other players, and thus, knowing the NPC locations is extremely important for competitive gameplay.

![](/images/npcs.jpg)

### 2. Methods
#### 2.1 Web-Scraping the Data / Data Cleaning
* In order to fine-tune the model, relevant data had to be scraped from the internet. All data except from weapon data was directly obtained from three main websites: oneesports.gg, dexerto.com, and dotesports. Weapon data was obtained from a well-known gaming analytics website called blitz.gg.
* The websites–from where most of the data were collected–follow a standard article format, with important information spread around the article. Data from these websites were then manually collected and organized before preparation for fine-tuning. Weapon data, on the other hand, was extremely hard to work with, as it was completely unstructured. Weapon information went through multiple iterations of data cleaning, from redacting irrelevant information, changing data types, to finally being structured into a Pandas dataframe.
![](/images/weapon_pandas.jpg)
<p align="center">
  <i>Raw data to Pandas DataFrame.</i>
</p>

#### 2.2 Data Preparation for Fine-Tuning
* Fine-tuning an OpenAI GPT model requires specific formatting of data. The data should follow the format of: context, prompt, and response. In addition to these formatting specifications, OpenAI recommends users to provide at least one hundred prompt-response pairs as the minimum amount of sufficient data for proper fine-tuning. To convert all the fine-tuning data into the proper format, the initial idea was to automate the data transformation by looping through all the data. However, this was not used as it would not capture the variety of prompts used by real users in terms of syntax. Instead, ChatGPT was used with the following prompt: ”Based on the provided data, generate 150 prompt response pairs with a lot of variety in wording.” The same context prompt was used for all pairs: ”You are a helpful assistant for Fortnite players, only providing answers to Fortnite-related questions.” Utilizing ChatGPT solved the problem of repetitive wording, and structured the data with the appropriate formatting for fine-tuning.

#### 2.3 Model Creation
* The chosen GPT model for fine-tuning was OpenAI’s gpt-4o-mini, which OpenAI describes as a balanced model that combines both performance and cost
efficiency. Model parameters were set to their automatically assigned values, except for n_epochs. The model was trained on 34,386 tokens, a batch size of 1, a learning rate of 1.8, and 8 epochs. An epoch describes the number of times the model sees the entire dataset. Too few epochs could lead to underfitting, while too many epochs could lead to overfitting. An epoch of 3 was the automatically chosen value—presumably because of the relatively small size of the training data—but was later changed to 8. The fine-tuned model had a training loss of 0.0505, as shown in the figure below.
![](/images/trainingloss.png)
<p align="center">
  <i>Training loss of fine-tuned model.</i>
</p>

### Results
* In order for the fine-tuned model to be considered helpful in the context of a gaming assistant, the most important metric is ***response accuracy***. Players should be able to receive reliable information about loot, NPCs, weapons, and locations. **The worst scenario in this context is model hallucination**, in other words, receiving factually-incorrect information from the AI assistant.
* An important parameter that can influence the quality of output is model temperature. This parameter ranges from 0 to 1, where lower values mean less randomness in response, while higher values mean more randomness in response. In the context of gaming, a lower temperature could be beneficial, as deterministic responses could result in more accurate output.

![](/images/model_temp.png)

  

* **The figure below shows that the fine-tuned model is indeed more knowledgeable about Fortnite in comparison to its regular counterpart.**

![](/images/modelcomparison.png)
<p align="center">
  <i>Response of the regular model (left) and factually-correct response of the fine-tuned model (right), with a temperature of 0.30.</i>
</p>

### Limitations
* Even though the fine-tuned model seems to be more knowledgeable about Fortnite, it is still not perfect. Even with a model temperature of 0.00, which in theory should prevent hallucinations, the fine-tuned model outputs factually incorrect information with confidence, as show in Figure 3 below. **The model incorrectly named the point of interest** (Ice Spice’s Crib should have been Ice Isle), **its location on the map** (center of the map should have been northwest corner of the map), **and the weapons’ names** (Drum Gun and Snowball Launcher should have been Assault Rifle and Grappler).

![](/images/hallucination.png)
<p align="center">
  <i>Factually-incorrect response of the fine-tuned model, with a temperature of 0.00.</i>
</p>


* Some mistakes have little impact on gameplay. For instance, providing the wrong name for a point of interest may not significantly affect the user’s experience. However, more critical errors, such as mislabeling weapons or giving incorrect locations, can be detrimental. For example, telling a user that a location is in the center of the map when it is actually in the northwest corner could lead to confusion and frustration, ultimately hindering gameplay.

### Next Steps
* A gaming AI assistant would certainly elevate users’ gaming experiences to the next level. However, we see that fine-tuning an LLM model comes with its own set of challenges. Next token prediction machine learning models such as the GPT model suffer from hallucinations. Before this fine-tuned LLM can be deployed and ultimately become a Voice AI Assistant, the model has to be fine-tuned even further to prevent hallucinations. Potential solutions include: embedding data into a searchable database, or incorporating a retrieval-augmented generation (RAG) architecture. Until these challenges are solved, the incorporation of LLMs in the field of gaming will remain a mere promising concept with no end product.

![](/images/fortnite_future.webp)

<a href="#top">Back to top</a>

# Project 2: What Exactly Makes Wine Taste Good? <a name="wine"></a>
### Introduction
* Wine preferences are very subjective as people tend to like different things. But is there a way to create an objectively good tasting wine with the help of machine learning? In this project, I aimed to find out what exactly makes wine taste good by analyzing the similarities and differences of 1600 wines. Objective features of the wine included things like: fixed acidity, residual sugars, pH level, alcohol content, etc. Each wine was rated by three different wine experts and the median of the ratings was used as the wine score which ranged from 0 (very bad) to 10 (excellent).
![](/images/wine_image.jpeg)

### Methods
* Four different regression models were used to compare results: ordinary least squares, ridge regression, lasso regression, and elastic net.
* Cross-validation was used for choosing the tuning parameters of all models except ordinary least squares.
* All code was written in Python, and packages used were: numpy and pandas for data manipulation, seaborn for data visualization, and scikit-learn for model training and cross-validation.
### Creating the Regression Models
#### 1. Ordinary Least Squares
* We want to identify the coefficients of a linear model relating wine quality to different features of the wine. Our predictors are all of the features of the wine, and our response variable is the subjective rating that each wine was given by the wine experts. The complete list of 11 features includes: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, and alcohol. The model was split 70/30 as train/test data.
#### 2. Ridge Regression
* To optimize our ridge regression model, we utilized the default leave-one-out cross validation, and inputted a list of ![](https://latex.codecogs.com/gif.latex?%5Calpha) values [0.1, 0.11, 0.12, ..., 2] where higher values of ![](https://latex.codecogs.com/gif.latex?%5Calpha) correspond to stronger regularization. The final ![](https://latex.codecogs.com/gif.latex?%5Calpha) value was 0.21 as shown by the graph below. We see that when ![](https://latex.codecogs.com/gif.latex?%5Calpha%3D0.21), the mean squared error is the lowest, and as ![](https://latex.codecogs.com/gif.latex?%5Calpha) increases past 0.21, the mean squared error increases rapidly. This shows that some regularization can help model performance, while too much regularization can reduce model performance.
![](/images/ridge.png)
#### 3. Lasso Regression
* For the lasso model, 5-fold cross-validation was used. We inputted a list of ![](https://latex.codecogs.com/gif.latex?%5Calpha) values [0.001, 0.002, 0.003, ..., 1] and the amount of penalization chosen was [$\alpha=0.001$](https://latex.codecogs.com/gif.latex?%5Calpha%3D0.001). We observe that in contrast to ridge regression, lasso regression gets rid of some features completely, as residual sugar as well as density are now both 0. This is because lasso (L1 regularization) is considered a more strict shrinkage operation, and leads to sparser models.
![](/images/lasso.png)
#### 4. Elastic Net
* In an elastic net model, there are 2 tuning parameters we need to consider when using cross-validation: the L1 ratio and ![](https://latex.codecogs.com/gif.latex?%5Calpha). 
* L1 ratio = 0 is ridge regression
* L1 ratio = 1 is lasso regression.
* Our model's L1 ratio chosen by cross-validation was 1, so in this case our elastic net model is the same as a lasso regression model.
### Results
#### Features
![](/images/wine%201.jpg)
#### Comparing Models on Test Data
![](/images/all%20models.png)

<a href="#top">Back to top</a>
# Project 3: Can We Use Machine Learning to Predict Primary Election Outcomes? <a name="election"></a>
### Introduction
* In this study, I explored the question of: "Can we use data to predict the results of a primary election?". I aimed to create a more complete representation of each candidate by feature-engineering and combining two data sets: FiveThirtyEight's Election Candidates data set and the Federal Election Commision's (FEC) campaign finance data set. By doing so, we were able to take advantage of many useful features like their demographics, finances, and endorsements.
* All code was written in Python and packages used included: numpy and pandas for data manipulation, seaborn and matplotlib for data visualization and scitkit-learn for machine learning.
![](/images/election_image_cover.jpg)
### Data Cleaning
* FiveThirtyEight's election candidates data set was merged with FEC's campaign finance data set. Each data set had a distinct naming convention, so we dropped middle names, suffixes, and nicknames and instead matched based on first name and last name. Certain features like date of transaction and candidates' identification numbers were dropped from the dataframe, as they did not provide any relevant and quantifiable information.
### Feature Engineering
* The Federal Election Commission data set contained some double-counted finance data. Some columns such as total receipts, total disbursements and candidate contributions were either combined or subtracted from each other to prevent double counting and collinearity of features. In addition, some binary endorsement data from the other data set such as "Endorsed by Trump?, "Endorsed by Sanders?", and "Obama Alum?" were one-hot-encoded in order to make model creation more efficient.
### Visualizing Trends
* How much would being endorsed by big names like Trump, Biden, Sanders, and Warren help candidates win a primary election? Do these big names often endorse or anti-endorse a lot of people? We also wanted to visualize the financial support that candidates received for their campaigns. How many individual contributions do candidates usually receive?
![](/images/endorsement%20image.jpg)
![](/images/election_image_2.png)
### Machine Learning
#### Decision Trees and Random Forests
* I utilized scikit-learn's decision tree classifier model as well as the random forest classifier model. I utilized a 70/30 split for train/test data. Because of the huge number of features from the combined data set, I changed the parameter "max depth" to set a limit to the number of nodes and prevent overfitting on our test data.
![](/images/decision%20tree%20and%20random%20forests.jpg)
#### Limitations of Our Models
* Although most features included in both Republican and Democratic data sets were the same, the Democratic data set had a handful of extra features that the Republican data set did not have. It included features such as: whether the candidate was white, a veteran, an elected official, a self-funder, had a STEM background, etc. These differences in features between the data sets might have helped or not helped the accuracy of our model. Although being a self-funder would probably mean the candidate is less known and has less support, we can't exactly quantify the impact of these features nor can we attribute the differences in results to these factors.
### Results
* The models were surprisingly accurate in predicting who will win a primary election. Both the random forest model and the decision tree model performed similarly in accuracy on test data. In order to get a more complete summary of our models' performances, we also looked at precision and recall which gives us a better understanding of the true positives in our predictions.
![](/images/model%20results.png) 

<br>

<a href="#top">Back to top</a>
# Project 4: The impact of COVID-19 on New York City's Public Transportation System <a name="transportation"></a>
### Introduction
* This project aimed to summarize and create a comprehensive report of the impact of COVID-19 on New York City's public transportation system. I focused on two main public transportation systems that millions of New Yorkers rely on daily: the subway and the bus. Some questions I explored were: "How has COVID-19 affected public transportation ridership?" and "What are factors that correlate with ridership numbers slowly going back to normal?"
![](/images/subway_cover.jpg)
### Methodology
* All the code was written in Python. Main packages include: numpy and pandas (for data cleaning and manipulation), matplotlib and seaborn (for data visualization), folium (for the interactive map), and scikit-learn (for machine learning). 
* Data were extracted mainly from the Metropolitan Transportation Authority website, and New York State Department of Transportation's website.
### Data Visualization
#### Subway and Bus Riderships
* Of all New Yorkers who commute, 39 percent use the subway, and 11 percent take the bus. With the issue of an executive order to close non-essential businesses in March of 2020, public transportation services in New York City remained open, but were reduced significantly. We can see the dip in both subway and bus ridership in the figures below. We can also see the gradual rise in ridership in the following months, as safety measures like masks, sanitation, and social distancing were introduced.
![](/images/subway.PNG)
![](/images/bus.PNG)
#### Correlation with COVID-19 Cases
* We also explored the potential correlation between public transportation ridership and COVID-19 cases. We wanted to see if there was an inverse correlation between ridership numbers and COVID-19 cases. For example, if people get sick with COVID-19, do they still go out and use public transportation? Or do they stay home and follow the stay-at-home mandates that were in place? We also wanted to explore if the sentiment remained constant from 2020 to 2021. Did people take the pandemic more seriously in the beginning? We can see this phenomenon, as in early 2020, a rise in COVID-19 cases correlated with a stark drop in ridership numbers. In contrast, a rise in COVID-19 cases in early 2021 didn't seem to have much impact on ridership numbers at all, as they're seen steadily increasing back to normal pre-pandemic levels.
![](/images/subway%20covid.PNG)
![](/images/bus%20covid.PNG)
#### Interactive Map
* I created an interactive map that showed us the busiest subway stations in
New York City in 2020. Clicking on the interactive map allows us to see each
station’s rank and ridership numbers. The 5 busiest subway stations ranked in
order are:
1. Times Square - 42 Street
2. Grand Central - 42 Street
3. 34th Street - Herald Square
4. 14th Street - Union Square
5. Fulton Street

![](/images/map.png)

<a href="#top">Back to top</a>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

# Project 5: Analyzing Player Sentiment on Fortnite's Changes to the XP System <a name="fortnite"></a>
### Introduction
* Fortnite’s recent changes to how players can get XP (experience points) was received with mixed opinions. With the purchase of the Battle Pass (that costs 950 V-Bucks or around $8), players receive in-game rewards according to their levels. Players have a chance to win all of the Battle Pass rewards by reaching level 200 by the end of the season. One way to ensure you progress through all the levels and receive all of the rewards is by completing quests. Until recently, players were given quests that could be completed throughout the season, whenever they wanted. But with the changes to the Weekly Quests, players now have exactly one week to complete these quests before they expire forever. This has caused a lot of backlash in the community and has left casual players feeling like they're missing out on a lot of XP and potential rewards, and that there’s not enough time to complete their weekly quests.
![](/images/fortnite_cover.png)
### Web Scraping the Data from Reddit
* On the Fortnite subreddit r/FortNiteBR, there was an interesting discussion thread titled: “Can We All Agree That Having 1 Week To Do Quests is Stupid?”. This post garnered tremendous attention, being one of the most upvoted discussion posts this year, with almost 6,000 upvotes. To better understand player sentiment and what people are saying about these recent changes, I decided to web scrape this whole post from Reddit, import it into a DataFrame in Python, and analyze this discussion through data.
### Sorting the Data
* Two ways we can explore the data are by filtering comments by “Best” and filtering comments by “Top”. These are what these Reddit terms mean:
1. Best: the highest upvote to downvote ratio. Basically means a lot of people agree with the comment, and there are almost no “dislikes”.
2. Top: the highest number of upvotes regardless of downvotes. 

* Although they sound similar, having both of these metrics in hand will be useful, as they tell us different things about each comment’s popularity.
* The column "upvotes" is sorting the comments by "Top", while the column "upvote ratio rank" is sorting the comments by "Best".
### Sentiment Analysis
* To better understand each comment’s sentiment, I utilized a natural language processing (NLP) package called TextBlob, to add these 2 metrics as columns to the DataFrame:
1. Polarity score: a score that ranges from -1 (which indicates negative sentiment) to 1 (which indicates positive sentiment).
2. Subjectivity score: a score that ranges from 0 (factual) to 1 (subjective). Subjective sentences generally refer to personal opinion, emotion, or judgment.
* Since this is a Reddit discussion thread, it’s expected that most comments will have high subjectivity scores.
![](/images/fortnite_sent.png)
### Data Visualizations
![](/images/fortnite_words_enhanced.png)
![](/images/fortnite_image_sentiment_1.png)
![](/images/fortnite_sentiment_2.png)
### Player Insights
* Reading through the best comments and most upvoted comments, we see that there are a few recurring topics that had great discussion, and interesting insights from players of various backgrounds.
#### 1. Fortnite shouldn't feel like a full-time job
* A recurring topic that came up in discussion was that casual players don't have time to play Fortnite every single day. They want to enjoy the game and earn their Battle Pass rewards without having to commit multiple hours every day to the game. 
* Adults make up a considerable part of the player base. They play Fortnite to unwind after a long day of working, and feel like these changes to the XP system negatively affect them. Here's what they had to say about it:
![](/images/fortnite_image_fulltime.png)
#### 2. Fear of Missing Out (FOMO) Business Model
* The FOMO monetization model incentivizes habitual playing and purchasing. Epic Games utilizes this business model extensively in Fortnite, for example: with limited-time cosmetic items, challenges, events, etc. Although this model has been extremely effective for player retention and monetization, it seems like players are fed up with it this time around. Here's what some people had to say about it:
![](/images/fortnite_image_fomo.png)
### Limitations
* Reddit is a great place to engage in discussion and learn something new. I learned so much by browsing the thread and got to understand many different perspectives and how players feel about the recent changes in Fortnite. One amazing thing about Reddit is that there's a lot of information and a lot of unfiltered opinions from many users. People can be genuine and express their true opinions because of the anonymity that Reddit provides.

* But anonymity is a double-edged sword, as there's a noticeable drawback to this: we don't know the background and age of all the users. We know that different groups of people have different experiences and play the game differently. Older players might play the game more casually and less frequently, while younger players might have more time to complete all their quests. This is something we should keep in mind when finding insights from the discussion and the data.

### Closing Thoughts
* We were able to engage in deep discussion and took a step forward in understanding player sentiment.  This project was very interesting and gave me a lot of new insights in better understanding people's different experiences and how they play Fortnite. I hope to bring my knowledge and passion for gaming, user research, and data analysis to the next opportunity that arises. Thanks for reading!

<br>

<a href="#top">Back to top</a>
# Full Code
## Project 2 Code  <a name="code2"></a>
![](/images/wine_proj_full.png)

<br>

<a href="#top">Back to top</a>
## Project 3 Code  <a name="code3"></a>
![](/images/DATA_102_Final_Project_Code-2-01.png)
![](/images/DATA_102_Final_Project_Code-2-02.png)
![](/images/DATA_102_Final_Project_Code-2-03.png)
![](/images/DATA_102_Final_Project_Code-2-04.png)
![](/images/DATA_102_Final_Project_Code-2-05.png)
![](/images/DATA_102_Final_Project_Code-2-06.png)
![](/images/DATA_102_Final_Project_Code-2-07.png)
![](/images/DATA_102_Final_Project_Code-2-08.png)
![](/images/DATA_102_Final_Project_Code-2-09.png)
<br>

<a href="#top">Back to top</a>
## Project 4 Code  <a name="code4"></a>
![](/images/NYC%20Subway%20Bus%20and%20Covid-19-1-1.png)
<br>

<a href="#top">Back to top</a>
## Project 5 Code  <a name="code5"></a>
![](/images/Fortnite%20Project-1.png)
![](/images/Fortnite%20Project-2.png)
![](/images/Fortnite%20Project-3.png)
![](/images/Fortnite%20Project-4.png)
![](/images/Fortnite%20Project-5.png)
![](/images/Fortnite%20Project-6.png)
![](/images/Fortnite%20Project-7.png)
![](/images/Fortnite%20Project-8.png)
<a href="#top">Back to top</a>
