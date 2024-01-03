#!/usr/bin/env python
# coding: utf-8

# # SVM: Face Recognition Application
# ## This notebook outlines the application of Support Vector Machine in the field of Face Recognition

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import fetch_lfw_people


# In[3]:


faces = fetch_lfw_people(min_faces_per_person=60)


# In[4]:


print(faces.target_names)


# In[5]:


print(faces.images.shape)


# In[6]:


plt.figure(figsize=(10,10))
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])


# In[7]:


from sklearn.svm import SVC


# In[8]:


from sklearn.decomposition import PCA as RandomizedPCA


# In[9]:


from sklearn.pipeline import make_pipeline


# In[10]:


pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)


# In[11]:


svc = SVC(kernel='rbf', class_weight='balanced')


# In[12]:


model = make_pipeline(pca, svc)


# In[13]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)


# In[15]:


model.fit(Xtrain, ytrain)


# In[16]:


yfit = model.predict(Xtest)


# In[17]:


fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# In[18]:


from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))


# ### Confusion Matrix

# In[19]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
plt.figure(figsize=(10,10))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[20]:


from sklearn.model_selection import GridSearchCV


# In[21]:


param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}


# In[22]:


grid = GridSearchCV(model, param_grid)


# In[23]:


get_ipython().run_line_magic('time', '')
grid.fit(Xtrain, ytrain)


# In[24]:


print(grid.best_params_)


# In[25]:


model = grid.best_estimator_


# In[26]:


yfit = model.predict(Xtest)


# In[27]:


fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# In[28]:


from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))


# In[29]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
plt.figure(figsize=(10,10))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');

