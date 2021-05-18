#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis For Stock Data

# ## Data Preparation

# In[1]:


''' install required libraries '''

# !pip install textblob
# !pip install nltk
# !pip install wordcloud
# !pip install tweepy
# !pip install langdetect

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import pandas as pd
import re ,string, csv

import tweepy # to access tweet API
from tweepy import OAuthHandler # for Authentication

from textblob import TextBlob #for Valance of Sentence(polarity)

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# nltk.download('all') # Installing All from NLTK library
from nltk.corpus import stopwords # For Removing Stop words like < the , an , is ,..etc >
n_words= stopwords.words('english') #specify english stop words only
n_words.append("rt") #append rt for stop word dictionary

from nltk.tokenize import word_tokenize # for Tokenizing the sentnces as tokens
from nltk.stem.porter import PorterStemmer # converting words to their root forms ,speed and simplicity
porter = PorterStemmer() #Create stemmer obejct

from nltk.stem import WordNetLemmatizer # also converting words to their actual root forms(noun , verb ,aobjective) ,but it slow
lemmatizer = WordNetLemmatizer() #Create lemmatizer obejct

from wordcloud import WordCloud,STOPWORDS #Look at Words with highest Frequency for expression

from langdetect import detect_langs # Detect language for each tweets 

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk import ngrams
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[2]:


# Reading Datasets
stocks=pd.read_csv('C:/Users/user/Desktop/Sentiment/stocks_cleaned.csv')
Data=pd.read_csv('C:/Users/user/Desktop/Sentiment/stockerbot-export.csv',error_bad_lines=False)


# In[3]:


Data.head()


# ### Exploratory data analysis

# In[4]:


Data.info()


# In[5]:


'''Convert Columns data types '''

# stockerbot["timestamp"] = pd.to_datetime(stockerbot["timestamp"])
Data["text"] = Data["text"].astype(str)
Data["url"] = Data["url"].astype(str)
Data["company_names"] = Data["company_names"].astype("category")
Data["symbols"] = Data["symbols"].astype("category")
Data["source"] = Data["source"].astype("category")
Data=Data.drop(columns=['id'])


# In[6]:


Data.info()


# In[7]:


''' Split Timestamp Column into Dates and times '''

Data[['dayofweek','month','day','time','timezone', 'year']] = Data.timestamp.str.split(expand=True)
Data[['hour','minute','second']] = Data.time.str.split(':',expand=True)
Data.head(2)


# In[8]:


''' Check for null values '''
Data.isnull().any() 


# There are a null values in Company Names Column

# In[9]:


''' Check for null values in Company names columns '''

print(f'null :{Data.company_names.isnull().sum()}')
Data[Data['company_names'].isnull()] 


# Only One Null Values , so not important for us to delete or not

# In[10]:


# Take a look at 10 Largest Source 
total_sources = Data["source"].value_counts()
print(f'Most sources:\n{total_sources.nlargest(10)}')
plt.figure(figsize=(15,5))
# total_sources.head(50).sort_values(ascending=False).plot(kind='bar') 


# In[11]:


# Take a look at 10 Largest symbols 
total_companies = Data["symbols"].value_counts()
print(f'Most companies:\n{total_companies.nlargest(10)}')
plt.figure(figsize=(15,5))
# total_companies.head(50).sort_values(ascending=False).plot(kind='bar') 


# In[12]:


len(Data.text)


# In[13]:


# Delete Unwanted Some Text 
Data=Data[Data["text"]!='btc']


# ### Pre-Processing Text

# In[14]:


# Define Clean Function to fix text
def Clean(text):

  # Frist converting all letters to lower case
  text= text.lower()
  
  # removing unwanted digits ,special chracters from the text
  text= ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", text).split()) #tags
  text= ' '.join(re.sub("^@?(\w){1,15}$", " ", text).split())
    
  text= ' '.join(re.sub("(\w+:\/\/\S+)", " ", text).split())   #Links
  text= ' '.join(re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"," ", text).split()) 
  text= ' '.join(re.sub(r'http\S+', '',text).split())
  
  
  text= ' '.join(re.sub(r'www\S+', '',text).split())
  text= ' '.join(re.sub("\s+", " ",text).split()) #Extrem white Space
  text= ' '.join(re.sub("[^-9A-Za-z ]", "" ,text).split()) #digits 
  text= ' '.join(re.sub('-', ' ', text).split()) 
  text= ' '.join(re.sub('_', ' ', text).split()) #underscore 
  
  # Display available PUNCTUATION for examples
  #for c in string.punctuation:
       #print(f"[{c}]")
  
  # removing stopwards and numbers from STRING library
  table= str.maketrans('', '', string.punctuation+string.digits)
  text = text.translate(table)
  
  # Split Sentence as tokens words 
  tokens = word_tokenize(text)
  
  # converting words to their root forms by STEMMING THE WORDS 
#   stemmed1 = [lemmatizer.lemmatize(word) for word in tokens] #Covert words to their actual root
  stemmed2 = [porter.stem(word) for word in tokens] # Covert words to their rootbut not actual
  
  # Delete each stop words from English stop words
#   words = [w for w in stemmed1 if not w in n_words] #n_words contains English stop words
  words = [w for w in stemmed2 if not w in n_words] #n_words contains English stop words

  text  = ' '.join(words)
    
  return text


# In[15]:


# Text Before Pre-processing
Data.text


# In[16]:


#Delete unwanted source form our text 
Data=Data[Data["source"] != "test5f1798"]


# In[17]:


# apply Clean Funsction to our Text
Data.text=[Clean(x) for x in Data.text]


# In[18]:


Data.text


# In[19]:


# Delete Unwanted Some Text 
Data=Data[Data["text"]!='btc']


# In[20]:


# Text after Pre-processing
Data.text


# In[21]:


''' Detect Emotions for each text Form TextBlob Library '''

detectEmotion=[]
detectPolarity=[]

for txt in Data.text:
    
    analysis=TextBlob(txt)
    Polarity=analysis.sentiment.polarity
    
    if Polarity  <0:
        emotion='2'  #Negative
    elif Polarity>0: 
        emotion='1'  #Positive
    else:
        emotion='0'  #Neutral
        
    detectEmotion.append(emotion)
    detectPolarity.append(Polarity)
    
# detectEmotion=pd.DataFrame()

Data['Polarity']=detectPolarity
Data['Emotion'] =detectEmotion


# In[22]:


# Data.head(3)


# In[23]:


# Data  = Data[Data['verified'] == True]


# In[24]:


#check for valid string only to detect languages

TextValid=[]

for i in range(len(Data)):
    TextValid.append(bool(re.match('^(?=.*[a-zA-Z])', Data.iloc[i,0])))
    
Data['valid']=TextValid
print(len(Data[Data['valid']==False]))
print(len(Data[Data['valid']==True]))


# In[25]:


# valid string only

Data=Data[Data['valid']==True]


# In[26]:


'''Detect languages for each text to filter into specific Lang'''

languages = []

# Loop over the sentences in the data and detect their language
for row in range(len(Data)):
    languages.append(detect_langs(Data.iloc[row, 0]))
    
# print('The detected languages are: ', languages) >>> ['en':'N']
languages = [str(lang).split(':')[0][1:] for lang in languages] 

# Assign the list to a new feature 
Data['language'] = languages


# In[27]:


# look at Lang detected from our text

Data['language'].value_counts()


# In[28]:


# len(Data)


# In[29]:


# We Only want to deal with english text for now , so we will filter data for EN Only

Data=Data[Data['language']=='en']


# In[30]:


# len(Data)


# In[31]:


Data=Data[['text','url','year','month','day','dayofweek','hour','minute','second','source','symbols','Polarity','Emotion','language','verified']]
Data.head(4)


# In[32]:


apple=Data[['text','year','month','day','Polarity','Emotion']][Data.symbols=='AAPL']
apple


# In[33]:


# Percentage of each Emotions for apple only

app_neutral   = apple['text'][ apple['Emotion'] == '0']
app_positive = apple['text'][ apple['Emotion'] == '1']
app_negative = apple['text'][ apple['Emotion'] == '2']

print(f' Percentage Positive: {len(app_positive)/len(apple)}\n Percentage Negetive: {len(app_negative)/len(apple)}\n Percentage Neutral : {len(app_neutral)/len(apple)}')


# In[34]:


# the below function will create a word cloud

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word  # double check for nay links
                                and not word.startswith('#')  # removing hash tags
                                and word != 'rt'  
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS, # using stopwords provided by Word cloud its optional since we already removed stopwords :)
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    # using matplotlib to display the images in notebook itself.
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
  


# In[35]:


print("Most Positive words Frequency")
wordcloud_draw(app_positive, 'white')
print("Most Negative words Frequency")
wordcloud_draw(app_negative)
print("Most Neutral words Frequency")
wordcloud_draw(app_neutral, 'white')


# In[36]:


# Percentage of each Emotions overall symbols

df_neutral   = Data['text'][ Data['Emotion'] == '0']
df_positive  = Data['text'][ Data['Emotion'] == '1']
df_negative  = Data['text'][ Data['Emotion'] == '2']


print(f' Percentage Positive: {len(df_positive)/len(Data)}\n Percentage Negetive: {len(df_negative)/len(Data)}\n Percentage Neutral: {len(df_neutral)/len(Data)}')


# In[37]:


print("Most Positive words Frequency")
wordcloud_draw(df_positive, 'white')
print("Most Negative words Frequency")
wordcloud_draw(df_negative)
print("Most Neutral words Frequency")
wordcloud_draw(df_neutral, 'white')


# In[38]:


# Save Dataset
Data.to_csv("MystockData.csv",index = False)


# ## Model

# In[39]:


def NgramModels(Model , txt, n):
    
    x_train, x_test, y_train, y_test = train_test_split(Data['text'], Data['Emotion'], test_size=0.2, random_state=50)
    
    vect      = CountVectorizer(max_features=1000 , ngram_range=(n,n))
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    model     = Model
    t0        = time.time()
    model.fit(train_vect, y_train)
    t1        = time.time()
    predicted = model.predict(test_vect)
    t2        = time.time()
    time_train= t1-t0
    time_pred = t2-t1
    
    accuracy  = model.score(train_vect, y_train)
    predicted = model.predict(test_vect)
    
    report = classification_report(y_test, predicted, output_dict=True)
    print("Models with " , n , "-grams :\n")
    print('********************** \n')
    print(txt)
    print("Training time: %fs; Prediction time: %fs \n" % (time_train, time_pred))
    print('Accuracy score train set :', accuracy)
    print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
    print('Positive: ', report['1'])
    print('Neutral : ', report['0'])
    print('Negative: ', report['2'])
    print('\n --------------------------------------------------------------------------------------------------- \n')


# In[40]:


def KNN_Ngram(n):
    
    x_train, x_test, y_train, y_test = train_test_split(Data['text'], Data['Emotion'], test_size=0.2, random_state=50)
    
    vect      = CountVectorizer(max_features=1000 , ngram_range=(n,n))
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    for k in [1,3,5,7,10]:

        model = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
        t0        = time.time()
        model.fit(train_vect, y_train)
        t1        = time.time()
        predicted = model.predict(test_vect)
        t2        = time.time()
        time_train= t1-t0
        time_pred = t2-t1

        accuracy  = model.score(train_vect, y_train)
        predicted = model.predict(test_vect)

        report = classification_report(y_test, predicted, output_dict=True)

        print("Models with " , n , "-grams :\n")
        print('********************** \n')
        print("Classification Report for k = {} is:\n".format(k))
        print("Training time: %fs ; Prediction time: %fs \n" % (time_train, time_pred))
        print('Accuracy score train set :', accuracy)
        print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
        print('Positive: ', report['1'])
        print('Neutral : ', report['0'])
        print('Negative: ', report['2'])
        print('\n -------------------------------------------------------------------------------------- \n')


# In[41]:


def TFIDFModels(Model,txt):
    
    x_train, x_test, y_train, y_test = train_test_split(Data['text'], Data['Emotion'], test_size=0.2, random_state=50)
    
    vect      = TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True)
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    model     = Model
    t0        = time.time()
    model.fit(train_vect, y_train)
    t1        = time.time()
    predicted = model.predict(test_vect)
    t2        = time.time()
    time_train= t1-t0
    time_pred = t2-t1
    
    accuracy  = model.score(train_vect, y_train)
    predicted = model.predict(test_vect)
    
    report = classification_report(y_test, predicted, output_dict=True)
    
    print(txt)
    print("Training time: %fs; Prediction time: %fs \n" % (time_train, time_pred))
    print('Accuracy score train set :', accuracy)
    print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
    print('Positive: ', report['1'])
    print('Neutral : ', report['0'])
    print('Negative: ', report['2'])
    print('\n -------------------------------------------------------------------------------------- \n')


# In[42]:


def KNN_TFIDF():
    
    x_train, x_test, y_train, y_test = train_test_split(Data['text'], Data['Emotion'], test_size=0.2, random_state=50)
    
    vect      = TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True)
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    for k in [1,3,5,7,10]:

        model = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
        t0        = time.time()
        model.fit(train_vect, y_train)
        t1        = time.time()
        predicted = model.predict(test_vect)
        t2        = time.time()
        time_train= t1-t0
        time_pred = t2-t1

        accuracy  = model.score(train_vect, y_train)
        predicted = model.predict(test_vect)

        report = classification_report(y_test, predicted, output_dict=True)

        print("Classification Report for k = {} is:\n".format(k))
        print("Training time: %fs ; Prediction time: %fs \n" % (time_train, time_pred))
        print('Accuracy score train set :', accuracy)
        print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
        print('Positive: ', report['1'])
        print('Neutral : ', report['0'])
        print('Negative: ', report['2'])
        print('\n -------------------------------------------------------------------------------------- \n')


# In[43]:


SupportVectorClassifier=svm.SVC(kernel='linear')

LogReg2=NgramModels(Model=LogisticRegression(),txt='Logistic Regression Model : \n ', n=2)
LogReg3=NgramModels(Model=LogisticRegression(),txt='Logistic Regression Model : \n ', n=3)

svm2=NgramModels(Model=SupportVectorClassifier ,txt='Support Vectoer Classifier Model : \n ', n=2)
svm3=NgramModels(Model=SupportVectorClassifier ,txt='Support Vectoer Classifier Model : \n ', n=3)

DecTree2=NgramModels(Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ', n=2)
DecTree3=NgramModels(Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ', n=3)

KNN2=KNN_Ngram(2)
KNN3=KNN_Ngram(3)


# In[44]:


SupportVectorClassifier=svm.SVC(kernel='linear')

print('Models with Tfidf Feature extraction Techniques : \n')
print('************************************************ \n')

LogReg=TFIDFModels(Model=LogisticRegression(),txt='Logistic Regression Model : \n ')
svm=TFIDFModels(Model=SupportVectorClassifier,txt='Support Vector Classifier Model : \n ')
DecTree=TFIDFModels(Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ')
knn_tfidf=KNN_TFIDF()


# In[47]:


idx = pd.MultiIndex.from_product([['2-grams', '3-grams', 'TFIDF'],['Accuracy %']],names=['FeatureExtraction', 'Metric'])
col = ['LogisticRegression', 'SupportVectorClassifier', 'DecisionTree', 'KNeighborsClassifier']

Result = pd.DataFrame('*', idx, col)
Result.LogisticRegression=['81','78','95']
Result.SupportVectorClassifier=['81','78','98']
Result.DecisionTree=['81','78','99']
Result.KNeighborsClassifier=['79','77','87']


# In[48]:


Result

