#!/usr/bin/env python
# coding: utf-8

# ## Predicting Seattle Car Accident Severity

# This notebook contains all the steps and transformations I performed for the feature selection for the Predicting of Traffic Accident Severity project. The dataset used is the one Coursera provides.

# Road traffic injuries are currently estimated to be the eighth leading cause of death across all age groups globally, and are predicted to become the seventh leading cause of death by 2030.
# 
# Analysing a significant range of factors, including weather conditions, drugs or alcohol involved, parked cars involved, among others, make possible that an accurate prediction of the severity of the accidents can be performed.
# 
# Governments should be highly interested in accurate predictions of the severity of an accident, in order to reduce the time of arrival and thus save a significant amount of people each year. Others interested could be private companies investing in technologies aiming to improve road safeness.

# In[3]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().system('pip install folium')

import folium
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
import itertools

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Set the seed
seed = 27912
np.random.seed(seed)


# ### Introduction 

# In[5]:


df_main = pd.read_csv('F:\Jorge\Final Capstone Project\Data_Collisions.csv', low_memory=False)
df_main.head()


# In[6]:


df_main.columns


# We can have a look to the shape of our dataframe:

# In[7]:


df_main.shape


# ### Data Wrangling

# As there are some columns with missing values, I'm going to count them in each column.
# 
# "True" represents a missing value

# In[8]:


missing_data = df_main.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# As there are 19 columns with missing values, it's time to decide how to deal with it.

# #### Frequency columns:
#     
# ADDRTYPE: Since there are 1926 missing values and only three possible values , I'm replacing the missing values with the most frequent one.
# 
# COLLISION TYPE: I'm replacing the missing values with the most frequent one.
# 
# JUNCTIONTYPE, UNDERINFL, WEATHER, ROADCOND, LIGHTCOND: The same method as the previous.
#     

# #### Drop columns:
# 
# INTKEY (missing more than the half of rows), EXCEPTRSNCODE, EXCEPTRSNDESC, INATTENTIONIND, PEDROWNOTGRNT, SPEEDING, SDOTCOLNUM, LOCATION (also mentioned in the columns x and y).   
#     
#     

# 
# 
# So, with this information established. Let's begin with the values meant to be replaced.
# 
# 

# In[9]:


mainfreq = df_main['ADDRTYPE'].value_counts().idxmax()

df_main['ADDRTYPE'].replace(np.nan, mainfreq, inplace=True)


# In[10]:


mainfreq = df_main['COLLISIONTYPE'].value_counts().idxmax()

df_main['COLLISIONTYPE'].replace(np.nan, mainfreq, inplace=True)


# In[11]:


mainfreq = df_main['JUNCTIONTYPE'].value_counts().idxmax()

df_main['JUNCTIONTYPE'].replace(np.nan, mainfreq, inplace=True)


# In[12]:


mainfreq = df_main['UNDERINFL'].value_counts().idxmax()

df_main['UNDERINFL'].replace(np.nan, mainfreq, inplace=True)


# In[13]:


mainfreq = df_main['WEATHER'].value_counts().idxmax()

df_main['WEATHER'].replace(np.nan, mainfreq, inplace=True)


# In[14]:


mainfreq = df_main['ROADCOND'].value_counts().idxmax()

df_main['ROADCOND'].replace(np.nan, mainfreq, inplace=True)


# In[15]:


mainfreq = df_main['LIGHTCOND'].value_counts().idxmax()

df_main['LIGHTCOND'].replace(np.nan, mainfreq, inplace=True)


# Now, let's drop the selected columns.

# In[16]:


df_main.drop(['INTKEY','EXCEPTRSNCODE','EXCEPTRSNDESC','INATTENTIONIND','PEDROWNOTGRNT','SPEEDING','SDOTCOLNUM','LOCATION'], axis=1, inplace=True)


# Now, let's review again the column values in case I've missed something.

# In[17]:


df_main.columns


# In[18]:


df_main.shape


# There looks like SEVERITYCODE and SEVERITYCODE are the same. If it's true, I'll have to drop one because of redudance.

# In[19]:


df_main['SEVERITYCODE'].equals(df_main['SEVERITYCODE.1'])


# In[20]:


to_drop = ['SEVERITYCODE.1']


# There are some columns that seem to have unique values for each row:
# 
# 1.OBJECTID
# 
# 2.INCKEY
# 
# 3.COLDETKEY
# 
# 4.REPORTNO
# 
# 5.SDOT_COLCODE
# 
# 6.ST_COLCODE
# 
# 7.SEGLANEKEY
# 
# 8.CROSSWALKKEY

# In[21]:


to_study = ['OBJECTID', 'INCKEY', 'COLDETKEY', 'REPORTNO', 'SDOT_COLCODE', 'ST_COLCODE', 'SEGLANEKEY', 'CROSSWALKKEY']

for i in to_study:
    print(i + ": {}".format(len(df_main[i].unique())))


# Since the dataframe has 194673 rows, the columns that have the same of rows with diferente values don't add valuable information.

# In[22]:


to_drop.extend(['OBJECTID','INCKEY','COLDETKEY','REPORTNO'])


# Now, there are 3 columns that appear to be descriptions
# 
# 1. SEVERITYDESC
# 2. SDOT_COLDESC
# 3. ST_COLDESC
# 
# So, I'm going to look into them with their corresponding code column.

# In[23]:


df_1 = df_main.pivot_table(index=['SEVERITYCODE','SEVERITYDESC'], aggfunc='size')
df_1


# In[24]:


df_2 = df_main.pivot_table(index=['SDOT_COLCODE','SDOT_COLDESC'], aggfunc='size')
df_2


# In[25]:


df_3 = df_main.pivot_table(index=['ST_COLCODE','ST_COLDESC'], aggfunc='size')
df_3


# We can see that each code has a unique description, therefore the code columns already give us the required information so we can drop the description columns.
# 
# IMPORTANT: These columns could be useful for visualization purposes but in this case I have decided to drop them.

# In[26]:


df_main.drop(['SEVERITYDESC','SDOT_COLDESC','ST_COLDESC'], axis=1, inplace=True)


# The next step will be check the columns of time and dates.

# In[27]:


df_main[['INCDATE','INCDTTM']].sample(5)


# We can see that both columns give us the same information but INCDTTM also stores the time of the collision so I'm going to drop INCDATE.

# In[28]:


to_drop.extend(['INCDATE'])


# In[29]:


df_main.drop(to_drop, axis=1, inplace=True)


# In[30]:


df_main.columns


# The next step is going to be checking the columns that are more likely to be categories:
# 
# 1. SEVERITYCODE
# 2. STATUS
# 3. ADRRTYPE
# 4. COLLISIONTYPE
# 5. JUNCTIONTYPE
# 6. UNDERINFL
# 7. WEATHER
# 8. ROADCOND
# 9. LIGHTCOND
# 10.HITPARKEDCAR

# In[31]:


df_main['SEVERITYCODE'].value_counts()


# In[32]:


df_main['STATUS'].value_counts()


# In[33]:


df_main['ADDRTYPE'].value_counts()


# In[34]:


df_main['COLLISIONTYPE'].value_counts()


# In[35]:


df_main['JUNCTIONTYPE'].value_counts()


# I'm going to replace this last value 'Unknown" by the most frequent value.

# In[36]:


most_freq = df_main['JUNCTIONTYPE'].value_counts().idxmax()
df_main['JUNCTIONTYPE'].replace('Unknown', most_freq, inplace=True)


# In[37]:


df_main['UNDERINFL'].value_counts()


# It can be seen that there are four unique values but we only need two of them to represent yes or no. So I'm going to convert 0 to N and 1 to Y.

# In[38]:


df_main.loc[df_main.UNDERINFL == '0', 'UNDERINFL'] = "N"
df_main.loc[df_main.UNDERINFL == '1', 'UNDERINFL'] = "Y"


# In[39]:


df_main['UNDERINFL'].value_counts()


# In[40]:


df_main['WEATHER'].value_counts()


# Again, I'm going to replace the Unknown value by the most frequent value.

# In[41]:


most_freq = df_main['WEATHER'].value_counts().idxmax()
df_main['WEATHER'].replace('Unknown', most_freq, inplace=True)


# In[42]:


df_main['ROADCOND'].value_counts()


# Again, I'm going to replace the Unknown value by the most frequent value.

# In[43]:


most_freq = df_main['ROADCOND'].value_counts().idxmax()
df_main['ROADCOND'].replace('Unknown', most_freq, inplace=True)


# In[44]:


df_main['LIGHTCOND'].value_counts()


# In[45]:


most_freq = df_main['LIGHTCOND'].value_counts().idxmax()
df_main['LIGHTCOND'].replace('Unknown', most_freq, inplace=True)


# In[46]:


df_main['HITPARKEDCAR'].value_counts()


# Let's change the type of some columns:

# In[47]:


to_categ = ['SEVERITYCODE', 'STATUS', 'ADDRTYPE', 'COLLISIONTYPE', 'JUNCTIONTYPE', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'HITPARKEDCAR']


# In[48]:


df_main[to_categ] = df_main[to_categ].astype('category') 
df_main[['SDOT_COLCODE']] = df_main[['SDOT_COLCODE']].astype('object')
df_main[['INCDTTM']] = df_main[['INCDTTM']].astype('datetime64')


# In[49]:


df_main.dtypes


# Let's check if there are any columns left with NaN values:

# In[50]:


df_main.columns[df_main.isna().any()].tolist()


# It can be seen that ST_COLCODE has some missing values, we are going to drop it because the column SDOT_COLCODE gives us almost the same information. Regarding the X and Y columns we are going to use them for visualization purposes and after that they will be dropped.

# In[51]:


df_main.drop(['ST_COLCODE'], axis=1, inplace=True)


# Let's now split the dataset into train and test.

# In[52]:


(df_main, df_test) = train_test_split(
    df_main,
    train_size=0.7, shuffle=True, random_state=seed)


# ### Data Visualization

# In[53]:


plt.figure(figsize=(8,6))
sns.countplot(x ='SEVERITYCODE', palette='Set2', data = df_main) 
plt.title('Severity Code Count', fontsize=18)
plt.xlabel('Severity Code', fontsize=16)
plt.ylabel('Count', fontsize=16)


# In[54]:


df_main["SEVERITYCODE"].value_counts(normalize=True)*100


# It can be seen that the data is not balanced 70% of the collisions are type 1 which means prop damage, and 30% type 2 injury.

# Let's see how the evolution of number of collisions through the years.

# In[55]:


fig, ax = plt.subplots(figsize=(20,5))
sns.countplot(df_main['INCDTTM'].dt.year, palette='mako', ax=ax)
ax.set_xlabel('Year', fontsize=18)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylabel('Collision Count', fontsize=18)
plt.title('Collision count through years', fontsize=18)


# We can see that the number of collisions have been decreasing since 2015. We can also see that there is a small amount of collisions in 2020, this is probably due to the actual pandemic situation.

# After that, let's study the amount of collisions throughout the day. To do that, let's extract the hour from the INCDTTM column:

# In[56]:


df_main['HOUR'] = df_main['INCDTTM'].dt.hour


# In[57]:


df_main[['HOUR', 'INCDTTM']].sample(5)


# Now, let's create a label for each time of the day:

# In[58]:


bins = [0,4,8,12,16,20,24]
labels = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
df_main['TIME'] = pd.cut(df_main['HOUR'], bins=bins, labels=labels, include_lowest=True)


# In[59]:


df_main[['TIME','HOUR']].sample(5)


# Once everything is checked we can drop the HOUR column created earlier:

# In[60]:


df_main.drop(['HOUR'], axis=1, inplace=True)


# Now we can plot the amount of collisions given the time of the day:

# In[61]:


def time_of_day_plot(df_main, title):
    '''
    Creates a countplot visualizing the data throughout the day 
    including the frequency.
    
        Parameters:
            df(DataFrame): Data to be visualized
            title(str): Title for the plot
    '''
    ncount = len(df_main['TIME'])
    plt.figure(figsize=(12,8))
    ax = sns.countplot(x='TIME', palette='Oranges', data=df_main)
    plt.title(title, fontsize=18)
    plt.xlabel('Time of the day', fontsize=18)

    # Make twin axis
    ax2=ax.twinx()
    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax.set_ylabel('Count', fontsize=18)
    ax2.set_ylabel('Frequency [%]', fontsize=18)
    ax.tick_params(axis="x", labelsize=15)

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom', fontsize=15) # set the alignment of the text

    ax2.set_ylim(0,100*ax.get_ylim()[1]/ncount)

    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    ax2.grid(None)


# In[62]:


time_of_day_plot(df_main, 'Distribution of Collisions throughout the day')


# It is clearly seen that the time of the day when most collisions occur are Late Night and Noon.

# Let's now study how the influence of alcohol and drugs affects collisions.

# In[63]:


plt.figure(figsize=(8,6))
sns.countplot(x ='UNDERINFL', palette='Set2', data = df_main) 
plt.title('Under Influence Count', fontsize=18)
plt.xlabel('Under Influence', fontsize=16)
plt.ylabel('Count', fontsize=16)


# We see that in the majority of the collisions that occur, the drivers where not under the influence of alcohol/drugs. Let's now see the distribution of collisions inlfuenced by alcohol/drugs throughout the day:

# In[64]:


influenced = df_main['UNDERINFL'] == 'Y'
influenced = df_main[influenced]


# In[65]:


time_of_day_plot(influenced, 'Distribution of Collisions influenced by alcohol or drugs')


# We can see that 45% of the collisions when the influence of alcohol/drugs is present are at Late Night and 23% are at Night. The behavior is as expected.

# Let's study the distribution of the HITPARKEDCAR column:

# In[66]:


plt.figure(figsize=(8,6))
sns.countplot(x ='HITPARKEDCAR', palette='Set2', data = df_main) 
plt.title('Hit Parked Car Count', fontsize=18)
plt.xlabel('Hit Parked Car', fontsize=16)
plt.ylabel('Count', fontsize=16)


# There are few collisions when a parked car is hit. Let's study when more parked cars are hit during the day:

# In[67]:


hit = df_main['HITPARKEDCAR'] == 'Y'
hit = df_main[hit]


# In[68]:


time_of_day_plot(hit, 'Distribution of Collisions when a parked car is hit')


# We can see that 20% of parked cars are hit in the Morning, this percentage decreases throughout the day but it suddenly increases in the Late Night maybe due to the influence of alcohol and drugs as we have seen.

# We can study if people influenced by alcohol/drugs tend to hit parked cars more:

# In[69]:


sns.set_palette(sns.color_palette('magma_r'))
tempdf = hit[(hit['UNDERINFL']=='Y')|(hit['UNDERINFL']=='N')]
fig, ax = plt.subplots(figsize=(7,7))
ax.pie(tempdf['UNDERINFL'].value_counts(), textprops={'color':'white', 'fontsize': 14}, autopct='%1.0f%%', explode=[0,0.1])

lgd = ax.legend(tempdf['UNDERINFL'].unique(),
          title='Under Alcohol/Drugs Influence',
          loc='upper center',
          bbox_to_anchor=(1, 0, 0.5, 1))
#plt.savefig('./plots/8.png')
plt.show()


# It is observed that most of the parked cars are hit by people with no influence of alcohol/drugs.

# Let's study the distribution of collisions when there is influence of alcohol/drugs and a parked car is hit.

# In[70]:


#Filter the dataset with only the rows where HITPARKEDCAR is Y and UNDERINFL is Y
hit_infl = (df_main['HITPARKEDCAR'] == 'Y') & (df_main['UNDERINFL'] == 'Y')
hit_infl = df_main[hit_infl]


# In[71]:


time_of_day_plot(hit_infl, 'Distribution of Collisions when influenced by alcohol/drugs and a parked car is hit')


# We can see that people under the influence of alcohol/drugs tend to hit parked cars at Late Night.

# Let's create some more plots to see the behavior of the columns left.

# In[72]:


plt.figure(figsize=(12,8))
sns.countplot(y ='COLLISIONTYPE', palette='husl', data = df_main) 


# In[73]:


plt.figure(figsize=(12,8))
sns.countplot(y ='JUNCTIONTYPE', palette='husl', data = df_main)


# In[74]:


fig, axs = plt.subplots(nrows=3, figsize=(15,15))
sns.countplot(y ='WEATHER', palette='husl', data = df_main, ax=axs[0]) 
sns.countplot(y ='ROADCOND', palette='husl', data = df_main, ax=axs[1]) 
sns.countplot(y ='LIGHTCOND', palette='husl', data = df_main, ax=axs[2])


# We can see that most of the collisions occur when the Weather is Clear the Road Condition is Dry and the Lighting Condition is Daylight.

# Let's create a map of Seattle to visualize where the collisions are placed.
# 
# IMPORTANT: the map creation has been commented because it adds several MB to the final notebook

# In[75]:


from folium import plugins

# simply drop whole row with NaN in "price" column
df_map = df_main.dropna(subset=['X','Y'], axis=0)

# reset index, because we droped two rows
df_map.reset_index(drop=True, inplace=True)
 
latitude = df_map['Y'].mean()
longitude = df_map['X'].mean()

# let's start with a clean copy of the map of Seattle
seattle_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the collisions in the dataframe
collisions = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng in zip(df_map.Y, df_map.X):
    folium.Marker(
        location=[lat, lng],
        icon=None
    ).add_to(collisions)

# display map
seattle_map


# ### Pre-Proccesing

# We are going to drop the columns X and Y as we said earlier. We are also going to drop INCDTTM because we have the TIME column created earlier:

# In[76]:


df_main.drop(['X','Y','INCDTTM'], axis=1, inplace=True)


# In[77]:


df_main.shape


# After that, we create X for the features and y for the class:

# In[78]:


(X, y) = (df_main.drop('SEVERITYCODE', axis=1), df_main['SEVERITYCODE'])


# In[79]:


to_encode = ['STATUS', 
             'ADDRTYPE', 
             'COLLISIONTYPE',
             'JUNCTIONTYPE',
             'UNDERINFL',
             'WEATHER',
             'ROADCOND',
             'LIGHTCOND',
             'HITPARKEDCAR',
             'TIME']

le = LabelEncoder()

for feat in to_encode:
    X[feat] = le.fit_transform(X[feat].astype(str))


# ### Modeling

# Then we split the dataset into train and validation:

# In[80]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)


# #### Decission Tree

# In[81]:


tree_model = DecisionTreeClassifier(criterion='entropy', max_depth = 4)
tree_model.fit(X_train, y_train)
predTree = tree_model.predict(X_val)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_val, predTree))


# We can plot the generated tree structure:
# 
# 

# In[82]:


fn=['STATUS','ADDRTYPE','COLLISIONTYPE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT','JUNCTIONTYPE','SDOT_COLDOE',
   'UNDERINFL','WEATHER','ROADCOND','LIGHTCOND','SEGLANEKEY','CROSSWALKKEY','HITPARKEDCAR','TIME']
cn=['prop damage', 'injury']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (40,20), dpi=300)

out = tree.plot_tree(tree_model,
               feature_names = fn, 
               class_names=cn,
               fontsize=15,
               filled = True);

for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(3)


# #### K-nearest neighbors
# 

# In[89]:


Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_val)
    mean_acc[n-1] = metrics.accuracy_score(y_val, yhat)

    
    std_acc[n-1]=np.std(yhat==y_val)/np.sqrt(yhat.shape[0])

mean_acc


# In[86]:


plt.figure(figsize=(10,6))
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.40, color='aquamarine')
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[87]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[88]:


kNN_model = KNeighborsClassifier(n_neighbors = 16)


# #### Logistic Regression
# 

# In[91]:


LR_model = LogisticRegression(C=0.01, solver='sag', max_iter=1000)
LR_model.fit(X_train, y_train)
predLR = LR_model.predict(X_val)
print("Logistic Regression's Accuracy: ", metrics.accuracy_score(y_val,predLR))


# ### Model Evaluation
# 

# We are going to use the test dataset defined earlier: df_test First of all we need to split it into X and y, and apply the transformations that were applied to the training data after the split.

# In[93]:


df_test['HOUR'] = df_test['INCDTTM'].dt.hour
bins = [0,4,8,12,16,20,24]
labels = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
df_test['TIME'] = pd.cut(df_test['HOUR'], bins=bins, labels=labels, include_lowest=True)
df_test.drop(['HOUR'], axis=1, inplace=True)

df_test.drop(['X','Y','INCDTTM'], axis=1, inplace=True)

(X_test, y_test) = (df_test.drop('SEVERITYCODE', axis=1), df_test['SEVERITYCODE'])

to_encode = ['STATUS', 
             'ADDRTYPE', 
             'COLLISIONTYPE',
             'JUNCTIONTYPE',
             'UNDERINFL',
             'WEATHER',
             'ROADCOND',
             'LIGHTCOND',
             'HITPARKEDCAR',
             'TIME']

le = LabelEncoder()

for feat in to_encode:
    X_test[feat] = le.fit_transform(X_test[feat].astype(str))


# Let's define the funtion that plots the confusion matrix we are going to use later:
# 
# 

# In[94]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)


# #### Decision Tree

# In[97]:


#Train the model on the training data
tree_model.fit(X, y)

#Make predictions on the test data
yhat_tree = tree_model.predict(X_test)

#Compute the different metrics
jaccard_tree = metrics.jaccard_score(y_test, yhat_tree)
f1_tree = metrics.f1_score(y_test, yhat_tree, average='weighted') 
acc_tree = metrics.accuracy_score(y_test, yhat_tree)

#Print the results
print("Tree model Accuracy Score", acc_tree)
print("Tree model Jaccard Score: ", jaccard_tree)
print("Tree model F1 Score: ", f1_tree)


# In[98]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_tree, labels=[1,2])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_tree))

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Prop damage(1)','Injury(2)'],normalize= False,  title='Confusion matrix')


# We can see that our tree model does a good job classifying the Prop damage samples, 40506 (TN, if we take the Prop damage class as negative) but its behavior is not good when classifying Injury samples, it classifies 14160 samples as Prop damage when they really are Injury (FN).

# #### K-nearest neighbors

# In[100]:


#Train the model on the training data
kNN_model.fit(X, y)

#Make predictions on the test data
yhat_kNN = kNN_model.predict(X_test)

#Compute the different metrics
jaccard_kNN = metrics.jaccard_score(y_test, yhat_kNN)
f1_kNN = metrics.f1_score(y_test, yhat_kNN, average='weighted') 
acc_kNN = metrics.accuracy_score(y_test, yhat_kNN)

#Print the results
print("K Nearest Neighbors model Accuracy Score", acc_kNN)
print("K Nearest Neighbors model Jaccard Score: ", jaccard_kNN)
print("K Nearest Neighbors model F1 Score: ", f1_kNN)


# In[101]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_kNN, labels=[1,2])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_tree))

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Prop damage(1)','Injury(2)'],normalize= False,  title='Confusion matrix')


# We can see that the behavior is similar to the one that the tree model offers, but we can see that the kNN model does a better job classifying the Injury class.

# #### Logistic Regression

# In[103]:


#Train the model on the training data
LR_model.fit(X, y)

#Make predictions on the test data
yhat_LR = LR_model.predict(X_test)
yhat_LR_prob = LR_model.predict_proba(X_test)

#Compute the different metrics
acc_LR = metrics.accuracy_score(y_test, yhat_LR)
jaccard_LR = metrics.jaccard_score(y_test, yhat_LR)
f1_LR = metrics.f1_score(y_test, yhat_LR, average='weighted') 
loss_LR = metrics.log_loss(y_test, yhat_LR_prob)

#Print the results
print("Logistic Regression model Accuracy Score", acc_LR)
print("Logistic Regression model Jaccard Score: ", jaccard_LR)
print("Logistic Regression model F1 Score: ", f1_LR)
print("Logistic Regression mode Log loss ", loss_LR)


# In[104]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_LR, labels=[1,2])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_tree))

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Prop damage(1)','Injury(2)'],normalize= False,  title='Confusion matrix')


# We can see that the behavior is similar to the ones offered by the tree and kNN models but this model is the worst when classifying the Injury class.

# ### Summary

# In[105]:


#Create lists with values
algorithms = ['Decision Tree', 'K Nearest Neighbors', 'Logistic Regression']
acc_total = [acc_tree, acc_kNN, acc_LR]
jaccard_total = [jaccard_tree, jaccard_kNN, jaccard_LR]
f1_total = [f1_tree, f1_kNN, f1_LR]
loss_total = ['','',loss_LR]

#Create the dictionary
d = {'Algorithm':algorithms, 'Accuracy':acc_total, 'Jaccard':jaccard_total, 'F1-score': f1_total, 'LogLoss': loss_total}

#Create and visualize the DataFrame
results = pd.DataFrame(d)
results.set_index('Algorithm', inplace=True)
results


# The model that I would choose is the K Nearest Neighbors model since it offers the most balanced predictions out of the three models proposed.

# In[ ]:




