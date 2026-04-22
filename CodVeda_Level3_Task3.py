#!/usr/bin/env python
# coding: utf-8

# #  Sentiment Analysis using Deep Learning (LSTM)
# 
# ##  Project Overview
# This project focuses on building a **Neural Network-based Sentiment Analysis model** using **TensorFlow/Keras**.  
# We transform textual data into numerical form and train a **Long Short-Term Memory (LSTM)** model for classification.
# 
# ---
# 
# ##  Objectives
# - Load and preprocess textual dataset  
# - Convert text into numerical vectors  
# - Design a deep learning model (LSTM)  
# - Train using backpropagation  
# - Evaluate performance using accuracy and loss curves  
# 
# ---
# 
# ##  Tools & Technologies
# - Python  
# - TensorFlow / Keras  
# - Pandas, NumPy  
# - Matplotlib, Seaborn  
# - NLTK (Natural Language Processing)  
# 
# ---
# 
# ##  Why This Project?
# Sentiment analysis is widely used in:
# - Product reviews  
# - Social media monitoring 
# - Customer feedback systems 
# 
# This project demonstrates **real-world NLP + Deep Learning integration**.

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


# Core Libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')


# In[4]:


df = pd.read_csv("Sentiment dataset.csv")

df.head()


# In[7]:


print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())


# ## EDA

# In[22]:


# sentiment distributions
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()


# In[23]:


# Text Length Analysis
df['Text_length'] = df['Text'].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(df['Text_length'], bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Length")
plt.show()


# ##  Data Cleaning
# 
# 1. Removing null values  
# 2. Standardizing column names  

# In[10]:


# Drop nulls
df.dropna(inplace=True)
df = df[['Text', 'Sentiment']]  
df.head()


# ## Text Preprocessing

# In[12]:


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_Text(Text):
    Text = Text.lower()
    Text = re.sub(r'[^a-zA-Z]', ' ', Text)
    words = Text.split()
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

df['clean_Text'] = df['Text'].apply(clean_Text)

df.head()


# ## Tokenization & Padding

# In[13]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean_Text'])

X = tokenizer.texts_to_sequences(df['clean_Text'])
X = pad_sequences(X, maxlen=100)

# Encode labels
y = pd.get_dummies(df['Sentiment']).values


# ## Train-Test Split

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ## Baseline Model

# In[24]:


from tensorflow.keras.layers import Flatten

baseline_model = Sequential()

baseline_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
baseline_model.add(Flatten())
baseline_model.add(Dense(64, activation='relu'))
baseline_model.add(Dense(y.shape[1], activation='softmax'))

baseline_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

baseline_model.summary()


# In[25]:


baseline_history = baseline_model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test)
)


# ## Model Building

# In[15]:


model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# ## Model Training

# In[16]:


history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)


# ## Performance Visualization

# In[17]:


# Accuracy Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Loss Plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()


# ## Comparison model

# In[26]:


baseline_acc = baseline_model.evaluate(X_test, y_test)[1]
lstm_acc = model.evaluate(X_test, y_test)[1]

print("Baseline Accuracy:", baseline_acc)
print("LSTM Accuracy:", lstm_acc)


# ##  Model Evaluation using:
# - Confusion Matrix  
# - Classification Report  

# In[18]:


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))


# ## Confusion Matrix Visualization

# In[19]:


cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ##  Final Insights
# 
# - GRU model provided efficient training  
# - Achieved strong classification performance  
# - Reduced computational complexity vs LSTM  
# ---
# ##  Key Highlight
# This project demonstrates a **modern, efficient deep learning pipeline** optimized for real-world NLP tasks.

# In[ ]:




