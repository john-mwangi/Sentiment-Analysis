#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sframe


# # Read product review data

# In[23]:


products = sframe.SFrame('../Data/amazon_baby.gl')


# In[33]:


pwd


# In[34]:


products.save('../Data/products.csv', format='csv')


# In[2]:


import pandas as pd


# In[2]:


pwd


# In[3]:


products = pd.read_csv('../Data/products.csv')


products.head()


# In[4]:


from collections import Counter


# In[5]:


products['word_count'] = products['review'].apply(lambda x: Counter(str(x).split(' ')))


# In[6]:


products.head()


# In[9]:


products['word_count2'] = products['review'].str.split().str.len()


# In[11]:


products.head()


# # Number of reviews per item

# In[11]:


# Pivot table
reviews = pd.crosstab(index=products['name'], columns='reviews')


# In[19]:


reviews


# In[13]:


# Sort the created crosstab in descending order
review = reviews.sort_values(by='reviews', ascending=False)


# In[18]:


reviews


# # Explore Vulli Sophie the Giraffe Teether

# In[55]:


giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[56]:


giraffe_reviews.head()


# In[22]:


len(giraffe_reviews)


# In[23]:


giraffe_ratings = pd.crosstab(index=giraffe_reviews['rating'], columns='count')


# In[24]:


giraffe_ratings = giraffe_ratings.sort_values(by='count', ascending=False)


# In[25]:


giraffe_ratings


# # Build a sentiment classifier

# In[7]:


#ignore 3* ratings
products = products[products['rating'] != 3]


# In[8]:


products['sentiment'] = products['rating'] >= 4


# In[9]:


products.head()


# In[10]:


def convt(sentiment):
    if sentiment == True:
        return 1
    else:
        return 0


# In[34]:


convt(True)


# In[37]:


products['sentiment'] = products['sentiment'].apply(convt)


# In[12]:


products.head()


# In[21]:


#add sentiment column
import numpy as np
products['sentiment'] = np.where(products['rating'] >= 4, 1, 0)


# In[22]:


products.head()


# # Training the sentiment classifier

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


train_data, test_data = train_test_split(products, train_size=0.8, random_state=0)


# In[15]:


print (train_data.shape, test_data.shape)


# In[25]:


train_data.shape


# In[26]:


test_data.shape


# # Train the classifier

# In[16]:


from sklearn.linear_model import LogisticRegression


# In[28]:

sentiment_model = LogisticRegression()


# In[17]:


from sklearn import linear_model


# In[18]:


# select Logistic Regression
sentiment_model = linear_model.LogisticRegression()


# In[19]:

from sklearn.feature_extraction import DictVectorizer


# In[20]:


dicVector = DictVectorizer()


# In[21]:


# this becomes our new X
train_data_wc_dic = dicVector.fit_transform(train_data['word_count'])


# In[22]:


sentiment_model.fit(train_data_wc_dic, train_data['sentiment'])


# # Evaluate the sentiment analysis model

# ## Using ROC curve

from sklearn.pipeline import make_pipeline


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


pipe = make_pipeline(DictVectorizer(), LogisticRegression())


# In[186]:


pipe.fit(train_data['word_count'], train_data['sentiment'])


# In[27]:


sent_pred = pipe.predict(test_data['word_count'])


# In[28]:


sent_pred

# In[29]:


from sklearn.metrics import roc_curve, auc


# In[38]:


# incorrect format
fpr, tpr, threshold = roc_curve(sent_pred, test_data['sentiment'])


# In[30]:


# correct format
fpr, tpr, thresholds = roc_curve(test_data['sentiment'], sent_pred)


# In[31]:


roc_auc = auc(fpr, tpr)


# In[33]:



# In[38]:


thresholds


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


plt.figure()


# In[39]:


plt.plot(fpr, tpr, color='green', label='ROC curve (area = %0.2f)' %roc_auc)  
plt.plot([0,1],[0,1], color='blue', linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receive Operating Characteristics')



from sklearn import metrics


# In[41]:


cnf_matrix = metrics.confusion_matrix(test_data['sentiment'], sent_pred)


# In[42]:


cnf_matrix


# In[190]:


Accuracy = metrics.accuracy_score(test_data['sentiment'], sent_pred)
Prediction = metrics.precision_score(test_data['sentiment'], sent_pred)
Recall = metrics.recall_score(test_data['sentiment'], sent_pred)


# In[54]:


print ('Accuracy: %s, Prediction: %s, Recall: %s' %(Accuracy, Prediction, Recall))


# In[49]:


print('Accuracy:',Accuracy,'Prediction:',Prediction,'Recall:',Recall)



giraffe_reviews.head()


# In[51]:


import numpy as np


# In[58]:


giraffe_reviews['predicted_sentiment'] = pipe.predict(giraffe_reviews['word_count'])


# In[59]:


giraffe_reviews.head()


giraffe_reviews['predicted_prob'] = pipe.predict_proba(giraffe_reviews['word_count'])[:,1]

giraffe_reviews.head()


# In[65]:


giraffe_reviews = giraffe_reviews.sort_values(by='predicted_prob', ascending=False)


# In[67]:


giraffe_reviews.head()


# In[76]:


giraffe_reviews.tail()


# In[84]:


giraffe_reviews.dtypes


# In[87]:


del giraffe_reviews['predicted_prob_2']


# In[88]:


giraffe_reviews.columns


# In[92]:


format(giraffe_reviews['predicted_prob'].iloc[-1], '0.15f')


# In[93]:


format(giraffe_reviews['predicted_prob'].iloc[-2], '0.15f')


# In[78]:


format(giraffe_reviews['predicted_prob'].iloc[0], '0.8f')


# In[89]:


# Top most review
giraffe_reviews['review'].iloc[0]


# In[90]:


# Worst review
giraffe_reviews['review'].iloc[-1]


giraffe_reviews = giraffe_reviews.sort_values(by='predicted_prob', ascending=False)


# In[132]:


giraffe_reviews.head()



giraffe_reviews['review'].iloc[0]


# In[248]:


giraffe_reviews['review'].iloc[1]


# # Bottom 2 reviews

# In[133]:


giraffe_reviews.tail()


# In[247]:


giraffe_reviews.iloc[-1]


# In[249]:


giraffe_reviews['review'].iloc[-1]


# In[250]:


giraffe_reviews['review'].iloc[-2]


# # Assignment

# In[95]:


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
for word in selected_words:
    products[word] = products['word_count'].apply(lambda x: x[word] if word in x else 0)


# In[96]:


products.head()


# In[98]:


products.to_csv('products_awesome.csv')


# In[70]:


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
for word in selected_words:
    products[word] = products['word_count'].apply(lambda x: x[word] if word in x else 0L).sum()


# In[71]:


products.head()


# In[101]:


train_data2, test_data2 = train_test_split(products, train_size=0.8, random_state=0)


# In[76]:


products.head()


# In[102]:


from sklearn.linear_model import LogisticRegression


# In[103]:


selected_words_model = LogisticRegression()   #initialise the model


# In[104]:


features = selected_words


# In[105]:


selected_words_model.fit(train_data2[features], train_data2['sentiment'])


# In[106]:


selected_words_model.coef_


# In[107]:


features


# In[91]:


sent_pred2 = selected_words_model.predict(test_data2[features])


# In[92]:


cnf_matrix2 = metrics.confusion_matrix(test_data2['sentiment'], sent_pred2)


# In[93]:


cnf_matrix2


# In[94]:


Accuracy = metrics.accuracy_score(test_data2['sentiment'], sent_pred2)
Prediction = metrics.precision_score(test_data2['sentiment'], sent_pred2)
Recall = metrics.recall_score(test_data2['sentiment'], sent_pred2)


# In[95]:


Accuracy, Prediction, Recall


# In[97]:


diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[98]:


diaper_champ_reviews.head()


# In[105]:


diaper_champ_reviews = diaper_champ_reviews.sort_values(by='review', ascending=False)


# In[108]:


selected_words


# In[117]:


sent_pred2


# In[119]:


test_data2


# In[120]:


diaper_champ_reviews['sentiment2'] = selected_words_model.predict(diaper_champ_reviews[features])


# In[121]:


diaper_champ_reviews.head()


diaper_champ_reviews['prob'] = selected_words_model.predict_proba(diaper_champ_reviews[features])[:,1]


diaper_champ_reviews = diaper_champ_reviews.sort_values(by='prob', ascending=False)


# In[125]:


diaper_champ_reviews.tail()


# In[99]:


products.head()


# In[100]:


products.shape


# In[ ]:


diaper

