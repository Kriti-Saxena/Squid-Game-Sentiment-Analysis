# Sentiment Analysis using VADER 

About the Dataset 
The datates was downloaded from [Kaggle](https://www.kaggle.com/datasets/deepcontractor/squid-game-netflix-twitter-data). It was collected using tweepy Python package to access Twitter API. The dataset has tweets from october 5th to oct 28th 2021, and follows #squidgames on twitter. 

For Analysis, the data was preprocessed to make it suitable for vader sentiment analysis. The code uses regular expressions to remove any special characters or emoticons. Foloowing that, the analysis does exploratory data analysis mainly through visualisation. 
For sentiment analysis, vader is used because it is more focused on social media and can give better sentiment analysis. 


# Import libraries 
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
stopword=set(stopwords.words('english'))
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string
import nltk
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
import textblob 
from textblob import TextBlob
```
This has textblob too, but i eneded up using just vader 

# Read and explore data 
```
data = pd.read_csv("tweets_v8.csv")
data.head()
data.info()
data.describe()
data.isnull().sum()
data['user_location'].value_counts()
```

# Structuring a new date column to study the volume of tweets everyday
```
print("Start Date:\t",data.date.sort_values().iloc[0])
print("End Date:\t",data.date.sort_values().iloc[-1])
def date_new(date):
    dt_format = '%Y-%m-%d %H:%M:%S%z'
    return datetime.strptime(date, dt_format).strftime('%b %d')

data['date_new'] = data['date'].apply(lambda x: parse_date(x))
```
# Dropping the original date column and Visualization 
```
#droping date column, we will be using the new date column further 
data.drop(['date'], axis=1, inplace = True)

# Visualization 
sns.countplot(x='date_new', data=data)
plt.xticks(rotation=70)
plt.tight_layout()
plt.xlabel("Date")
plt.ylabel("Number of tweets")
sns.set(rc={"figure.figsize":(8, 5)})
```
```
sns.countplot(x='user_verified', data=data)
plt.xticks(rotation=70)
plt.tight_layout()
plt.xlabel("Date")
plt.ylabel("Number of tweets")
sns.set(rc={"figure.figsize":(8, 5)})
```

# Preprocessing the data 
```
# removing some emoji's and other special characters from the dataset
data = data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
data.head()
```

```
text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```
since it is considering https as one the most common word, we will clean the text column for a clear further analysis

# cleaning the data with regular expression
```
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["text"] = data["text"].apply(clean)
```

```
text = data['text'].values 

wordcloud2 = WordCloud().generate(str(text))

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud2)
plt.axis("off")
plt.figure( figsize=(15,10))
plt.show()
```

```
# Establishing polarity and subjectivity for the dataset 
data[['polarity', 'subjectivity']] = data['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
data.head()
```

```
def sentiment_scores(text):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(clean(text))
    if sentiment_dict['compound'] >= 0.05 :
        return "Positive"
    elif sentiment_dict['compound'] <= -0.05 :
        return "Negative"
    else :
        return "Neutral"
```
# vader sentiment analysis 
```
data['vader_Sentiment']=data['text'].apply(lambda x: sentiment_scores(clean(x)))
```
# Visualization 
```
fig = plt.figure(figsize=(8,6))
ax = sns.countplot(x=data['vader_Sentiment'])
total = len(data['vader_Sentiment'])
for p in ax.patches:
    x = p.get_x() + p.get_width() / 2 - 0.1
    y = p.get_y() + p.get_height() + 500
    ax.annotate(p.get_height(), (x, y), size = 10)
sns.set_palette("pastel")
plt.xlabel('Sentiment')
plt.title('VADER Sentiment')
plt.show()
```

From the analysis it is seen that squid games had more of a neutral sentiment analysis followed by positive and negative the least. This result is helpful for the makers to track reactions on twitter for each of the episodes. 

###### Learning from the project

Getting hands dirty in real data always exposes one to new skills, and approaches for a problem, and in this case, to new packages, and codes. This was my second project on python, I was able to apply what I learnt in theory about NLP and sentiment analysis to action. I got exposed to new concepts such as regular expressions in python, and wordcloud visulaisation. 
