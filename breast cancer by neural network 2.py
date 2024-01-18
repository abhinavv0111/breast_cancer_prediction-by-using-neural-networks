#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 


# In[2]:


import sklearn.datasets


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import pandas as pd


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[7]:


print(breast_cancer_dataset)


# In[8]:


data_frame = pd.DataFrame(breast_cancer_dataset.data,columns = breast_cancer_dataset.feature_names)


# In[9]:


data_frame.head()


# In[10]:


data_frame['label'] = breast_cancer_dataset.target


# In[11]:


data_frame.tail()


# In[12]:


data_frame.info()


# In[13]:


data_frame.describe()


# In[14]:


data_frame.isnull().sum()


# In[15]:


data_frame['label'].value_counts()


# In[16]:


data_frame.groupby('label').mean()


# In[17]:


X = data_frame.drop(columns = 'label', axis = 1)
Y = data_frame['label']


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.2, random_state=2)


# In[21]:


print(X.shape,X_train.shape,X_test.shape)


# # standardize the data

# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()


# In[24]:


X_train_std = scaler.fit_transform(X_train)


# In[25]:


X_test_std = scaler.fit_transform(X_test)


# In[26]:


print(X_train_std)


# In[27]:


print(X_test_std)


# In[28]:


import tensorflow as tf 


# In[29]:


from tensorflow import keras


# In[30]:


tf.random.set_seed(3)


# In[31]:


model = keras.Sequential([
                           keras.layers.Flatten(input_shape=(30,)),  ## INPUT SHAPE NO IS EQUAL TO COLUMSOR FEATURES IN DATASETS
                           keras.layers.Dense(20, activation='relu'),
                           keras.layers.Dense(2,activation='sigmoid') ##the neuron given in op layer is equL TO THE NO OF CLASSES HAVE HRE 0 ND 1 TWO CLASSES THEREFORE GIVE 2 NEURON
])


# In[32]:


## COMPILING THE NEURAL NETWORK


# In[33]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[34]:


##training the neural network


# In[35]:


history = model.fit(X_train_std , Y_train , validation_split = 0.1, epochs = 10)## take 10% ofdata in traain in 10 rounds


# In[36]:


# Evaluate the model on the test set


# In[37]:


test_loss, test_accuracy = model.evaluate(X_test_std, Y_test)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')


# In[38]:


# Plot training history


# In[39]:


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[40]:


# Make predictions on the test set
predictions = model.predict(X_test_std)


# In[41]:


# Print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix


# In[42]:


# Make predictions on the test set
predictions = model.predict(X_test_std)


# In[43]:


# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)


# In[44]:


print("Classification Report:\n", classification_report(Y_test, predicted_labels))
print("Confusion Matrix:\n", confusion_matrix(Y_test, predicted_labels))


# In[ ]:




