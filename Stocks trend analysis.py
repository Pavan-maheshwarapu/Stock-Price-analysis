#!/usr/bin/env python
# coding: utf-8

# # Stock Market Trend Analysis Model
# Building a model to analysis the historical data of an index and predict the future.
# This is the analysis on nifty 50 from 1990 January 1 to 2024 February 

# In[3]:


import pandas as pd


# In[4]:


path = '‪C:\MLprojects\\nifty.csv'
nifty = pd.read_csv(path.strip('\u202a'))#text editor is creating some unicode before the path. so need to remove that code
nifty.Close = nifty.Close.astype(int)#changing data type of close price to int


# In[5]:


nifty.head()


# In[6]:


nifty.plot.line(y='Close', x='Tradingday')


# In[7]:


nifty['Tomorrow'] = nifty['Close'].shift(-1)
#Add a column to the right of the close price with tomorrow's prices


# In[8]:


nifty.head()


# In[9]:


nifty['Target'] = (nifty['Tomorrow'] > nifty['Close']).astype(int)
#gives a boolean values of whether the tomorrow price is greater than today's closing price or not.
nifty.head(10)


# In[10]:


from sklearn.ensemble import RandomForestClassifier as rfc

model = rfc(n_estimators=100, min_samples_split=100, random_state=1)
#estimators is the no. of individual decision trees we want to train
# min sample split protect against overfitting, higher the less accurate the model would be but the lesser it will overfit
#random_state to get the same results upon repeated running of the  model


train = nifty.iloc[:-100]
test = nifty.iloc[-100:]

predictors = ['Close', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])


# In[11]:


from sklearn.metrics import precision_score as ps
#evaluating the performance 

preds = model.predict(test[predictors])


# In[12]:


preds = pd.Series(preds, index=test.index)
preds.head(10)


# In[13]:


ps(test['Target'], preds)


# not a good score to use in real life. so there is need to make the model better

# In[14]:


combined = pd.concat([test['Target'], preds], axis=1)


# In[15]:


combined.plot()


# In[16]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined


# In[17]:


#
def backtest(data, model, predictors, start=2500, step=250): #taking 10 years of data and predict the values for 11th year
    all_predictions =[]
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[18]:


predictions = backtest(nifty, model, predictors)


# In[19]:


predictions["Predictions"].value_counts()


# In[20]:


ps(predictions['Target'], predictions['Predictions'])


# In[21]:


predictions["Target"].value_counts()/predictions.shape[0]


# In[22]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = nifty.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    nifty[ratio_column] = nifty["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    nifty[trend_column] = nifty.shift(1).rolling(horizon).sum()['Target']
    
    new_predictors += [ratio_column, trend_column]
    


# In[23]:


nifty = nifty.dropna()
nifty


# In[24]:


model = rfc(n_estimators=200, min_samples_split=50, random_state=1)


# In[33]:


#making the more accurate using predict_proba
def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:,1] # gives the probability of price going up 
    preds[preds >= 0.7] = 1 # marks as price increasing only if the probability is more than 60%
    preds[preds < 0.7] = 0
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined


# In[34]:


predictions = backtest(nifty, model, new_predictors)


# In[35]:


predictions['Predictions'].value_counts()


# In[36]:


ps(predictions['Target'], predictions['Predictions'])


# In[ ]:





# In[ ]:




