# USA Presidential Debates 2020 : EDA, Sentiment Analysis and Predictive Modelling

<!-- ![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/debate.png) -->

### US President Donald Trump and President-Elect Joe Biden had their chance to challenge each other face to face twice during the Presidential Debates. In this project we're trying to analyze and visualize a dataset that contains the transcripts of the debates.

## Code and Resources Used :
- Python : 3.8.5
- Libraries : pandas, seaborn, wordcloud, bs4, re, nltk, sklearn
- Data : [Kaggle](https://www.kaggle.com/headsortails/us-election-2020-presidential-debates), additional data scraped from [Factba.se](https://factba.se/transcripts) and [Rev](https://www.rev.com/blog/transcripts/all-transcripts)
  
## Data Cleaning
### Upon initial analysis of the data obtained from Kaggle we can identify a number of data cleaning and pre-processing steps required.
- ### Null values : The 'minute' column of the first debate consists a null value. Upon inspection we find that it represents the start of the second segment and so we replace it with '00:00'.
- ### Inconsistent speaker names : There are inconsistencies in the speaker names such as Chris Wallace being represenred as 'Chris Wallace' and 'Chris Wallace : ', President Trump being represented as 'President Donald J. Trump', 'President Trump' and 'Donald Trump'. All of these are normalised to 'Donald Trump'.
- ### Inconsistent timeframe : The 'minute' column is in string form. To perform analysis we will convert it to seconds spoken by each candidate.

## Data Analysis
### We will start off with some basic EDA. 
### We will first find the candidate who spoke for the longest time in one go and what did he speak.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/screenshots/long_speak_time.png) 

### We will now look at the vocabulary size for both candidates.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/screenshots/vocab.png)

### We will now look at the total time spoken by the candidates and the moderators across both debates.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/screenshots/time_spoken.png)
### We can see that it was a neck to neck debate between both the candidates.
### We will now look at the total words spoken by the candidates and the moderators across both debates.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/screenshots/words_spoken.png)
### Interesting insight - Donald Trump spoke for a total of 30 seconds less than Joe Biden but spoke 1000 more words.

## Word Frequency :
### Let us look at the most frequent words used by the candidates and the moderators. For better analysis, we've performed some text processing operations such as :
- ### Remove stopwords
- ### Remove punctuations
- ### Remove numbers
- ### Convert words to lowercase
- ### Converting contractions to words

### Donald Trump
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/trump_freq.jpg)
### Joe Biden
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/biden_freq.jpg)
### Moderators ( Chris Wallace and Kristen Welker )
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/mod_freq.jpg)

## Bigram Frequency :
### A bigram is a pair of consecutive written units such as letters, syllables, or words. We will now look at the most frequent bigrams used by the candidates.
### Donald Trump
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/trump_bg.jpg)
### Joe Biden
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/biden_bg.jpg)

## WordClouds
### Let us create some word clouds!
### Donald Trump
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/trump_wordc.jpg)
### Joe Biden
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/biden_wordc.jpg)

## Flow of the debate
### Hardly a minute went by during the debates without one of the candidates angrily interrupting the other, whether on the coronavirus pandemic, the Supreme Court, the economy or anything else, including each other’s families. 
### “Will you shut up, man?” Biden snapped at Trump at one point.
### A good way to visualize the number of interruptions was by plotting heatmaps of the flow of the debates.
### First Presidential Debate :
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/first.jpg)
### Second Presidential Debate :
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/second.jpg)
### Vice Presidential Debate :
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/vp_map.jpg)
### These heatmaps give us an idea about the high amount of interruptions and cross talking during the first debate. In the second presidential debate, the mute button, or at least the threat of it seemed to work as Donald Trump and Joe Biden were more restrained.

## Sentiment Analysis :
### We will use the ``` SentimentIntensityAnalyzer ``` from ```nltk``` package to calculate the sentiments of the sentences spoken by both the candidates.
### We use the compound score to measure the sentiments which ranges from -1 (Most Negative) to +1 (Most Positive).
### After calculating the sentiment (Positive, Neutral, Negative) we can visualize it.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/plots/sentiment.jpg)

## Model Building
### Now we will build a machine learning model that will train on the debate transcript data, town hall speeches data and the data extracted from factba.se and rev.com. The model will then predict whether a given quote could be spoken by Donald Trump or Joe Biden.
### Data has been scraped using BeautifulSoup. After scraping the blocks of text, they've been split into sentences and given the appropriate 'speaker' label and then converted to a dataframe.
### ```train_test_split``` from ```sklearn``` is used to split the data into training and test data in the ratio 80:20.
### Next we use ```TfidfVectorizer``` which transforms text to feature vectors that can be used as input to estimator. tf-idf creates a set of its own vocabulary from the entire set of text. Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus.
### Then we use ```LogisticRegression``` to train the model. We can compute the ```confusion_matrix``` to find the accuracy of the model.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/screenshots/conf_matrix.png)
### The confusion matrix gives us an accuracy of 85.83%.
### Let us check the model's predictions for some different Joe Biden quotes.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/screenshots/biden_quotes.png)
### Let us check the model's predictions for some different Donald Trump quotes.
![](https://raw.githubusercontent.com/ritik-k/USA_election_debate/master/images/screenshots/trump_quotes.png)

***
## Usage:
### This project is best viewed in a notebook viewer, which can be accessed here:
- ### EDA, Word Frequency plots, Bigram Frequency plots, Word clouds, Sentiment Analysis - [here](https://nbviewer.jupyter.org/github/ritik-k/USA_election_debate/blob/master/main.ipynb)
- ### Debate Flow Heatmaps - [here](https://nbviewer.jupyter.org/github/ritik-k/USA_election_debate/blob/master/heatmaps.ipynb)
- ### Web Scraper - [here](https://nbviewer.jupyter.org/github/ritik-k/USA_election_debate/blob/master/scraper.ipynb)
- ### ML Model - [here](https://nbviewer.jupyter.org/github/ritik-k/USA_election_debate/blob/master/model.ipynb)