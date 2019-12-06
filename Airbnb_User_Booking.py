#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports several basic packages
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# ignore the warnings
import warnings 
warnings.filterwarnings("ignore")

# import the package of heterogenous classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV


# In[3]:


# input the data including train set, test set, countries set, age_gender set and the session set
train_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_2.csv')
test_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_users.csv')
countries = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/countries.csv')
age_gender = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/age_gender_bkts.csv')
sessions = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/sessions.csv')


# # Step1: Exploratory Data Analysis

# In[179]:


print('There are', train_users.shape[0], "users in the training set and", test_users.shape[0], "in the test set.")
print("In sum we have", train_users.shape[0] + test_users.shape[0], "users.")

# preview the train set
print("Top View of train_users\n")
print(train_users.head())
print(train_users.info())
print('The shape of train_users:', train_users.shape)
print()

# previes the test set
print("Top View of test_users\n")
print(test_users.head())
print(test_users.info())
print('The shape of test_users:', test_users.shape)
print()

# previes the countries set
print("Top View of countries\n")
print(countries.head())
print(countries.info())
print('The shape of countries:', countries.shape)
print()

# previes the age_gender set
print("Top View of age_gender\n")
print(age_gender.head())
print(age_gender.info())
print('The shape of age_gender:', age_gender.shape)
print()

# previes the sessions set
print("Top View of sessions\n")
print(sessions.head())
print(sessions.info())
print('The shape of sessions:', sessions.shape)
print()


# In[180]:


# Several information of train_users
train_users.describe()


# In[181]:


# Several information of test_users
test_users.describe()


# In[182]:


# Several information of sessions
sessions.describe()


# In[183]:


# missing values in train_users
train_users.isnull().any()


# In[184]:


# missing values in test_users
test_users.isnull().any()


# In[185]:


# missing values in sessions
sessions.isnull().any()


# In[186]:


# Date visualization
dt = train_users
dt['date_account_created'] = pd.to_datetime(dt['date_account_created'])

plt.figure(figsize = (20, 10))
dt[dt['country_destination'] != 'NDF']['date_account_created'].value_counts().plot(kind = 'line', color = 'deepskyblue')
plt.title('AxesSubplot with time series')
plt.xlabel('Date account created')
plt.ylabel('Number of Airbnb users')


# In[187]:


train_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_2.csv')


# In[5]:


# define the function to obtain the year of 'date_first_booking'
def get_year(date):
    if date == date: 
        return str(int(str(date)[:4]))
    return date


# In[6]:


# define the function to obtain the month of 'date_first_booking'
def get_month(date):
    if date == date: 
        return str(int(str(date)[5:7]))
    return date


# In[7]:


# extract the year and month of 'date_first_booking'
from datetime import datetime
    
train_users['year_first_booking'] = train_users['date_first_booking'].apply(get_year)
train_users['month_first_booking'] = train_users['date_first_booking'].apply(get_month)
test_users['year_first_booking'] = test_users['date_first_booking'].apply(get_year)
test_users['month_first_booking'] = test_users['date_first_booking'].apply(get_month)

# fill NaN using the median 
train_users['year_first_booking'].fillna(train_users['year_first_booking'].median(), inplace = True)
train_users['month_first_booking'].fillna(train_users['month_first_booking'].median(), inplace = True)
test_users['year_first_booking'].fillna(test_users['year_first_booking'].median(), inplace = True)
test_users['month_first_booking'].fillna(test_users['month_first_booking'].median(), inplace = True)

# convert type to integer
train_users['year_first_booking'] = train_users['year_first_booking'].astype(int)
train_users['month_first_booking'] = train_users['month_first_booking'].astype(int)


print('Top View of train_users:')
print(train_users.head())
print()

print('Top View of test_users:')
print(test_users.head())


# In[8]:


# define the function to obtain the booking information of each user
# 'NDF' stands for no destination found, namely no booking
def get_booking(country_destination):
    if country_destination == 'NDF':
        country_destination = 'No booking'
    else: 
        country_destination = 'Booking'
    return country_destination


# In[217]:


# booking or not booking visualization

# frequency of booked vs. no-booking users of each year
# NOTICE that in year 2014, and 2015 there wasn't 'no-booking'
train_users['Booked_or_not'] = train_users['country_destination'].apply(get_booking)
plt.figure(figsize = (12, 8))
Airbnb_users_booked_year = sns.countplot(x = 'year_first_booking', hue = 'Booked_or_not', data = train_users, palette = 'deep', order = [2010,2011,2012,2013,2014,2015])
plt.title('Number of booking vs. non_booking Airbnb users every year')
plt.xlabel('Year')
plt.ylabel('Number of booking and non_booking Airbnb users')


# country visualization
# Plot the frequency for every country_destination 
plt.figure(figsize = (12, 8))
Airbnb_users_country = sns.countplot(x = 'country_destination', data = train_users[train_users['country_destination'] != 'NDF'], palette = 'deep')
plt.title('Number of Airbnb users visiting each country from 2010-2015')
plt.xlabel('Country_destination')
plt.ylabel('Number of Airbnb vistiors')

# show the exact number of each country in the histogram
for k in Airbnb_users_country.patches:
    Airbnb_users_country.annotate('{:.1f}'.format(int(k.get_height())), (k.get_x() + 0.1, k.get_height()))

# Plot the distance of US and other countries
plt.figure(figsize = (12, 8))
Airbnb_users_country_dis = sns.barplot(x = 'country_destination', y = 'distance_km', data = countries)
plt.title('Distance between US and other countries')
plt.xlabel('Country_destination')
plt.ylabel('Distance_km')

# show the exact number of each country in the histogram
for k in Airbnb_users_country_dis.patches:
    Airbnb_users_country_dis.annotate('{:.1f}'.format(int(k.get_height())), (k.get_x() + 0.1, k.get_height()))

    
# Plot the frequency for every country_destination 
plt.figure(figsize = (12, 8))
Airbnb_users_country_year = sns.countplot(x = 'year_first_booking', hue = "country_destination", data = train_users[train_users['country_destination'] != 'NDF'], palette = 'deep')
plt.title('Number of Airbnb users visiting each country every year')
plt.xlabel('Year')
plt.ylabel('Number of Airbnb vistiors')


# In[218]:


# gender visualization

# ignore the warnings
import warnings 
warnings.filterwarnings("ignore")
  
# frequency of country_destination of each gender
plt.figure(figsize = (12, 8))
Airbnb_users_country_gender = sns.countplot(x = 'gender', hue = 'country_destination', data = train_users[train_users['country_destination'] != 'NDF'], palette = 'deep', order = ['FEMALE', 'MALE'])
plt.title('Number of Airbnb users visiting each country of disparate genders')
plt.xlabel('Gender')
plt.ylabel('Number of Airbnb users visiting each country')


# frequency of booked vs. no-booking users of each gender
plt.figure(figsize = (12, 8))
Airbnb_users_booked_gender = sns.countplot(x = 'gender', hue = 'Booked_or_not', data = train_users)
plt.title('Number of booked vs. non_booking Airbnb users of disparate gender')
plt.xlabel('Gender')
plt.ylabel('Number of booked and non_booking Airbnb users')

# show the exact number of each country in the histogram
for k in Airbnb_users_booked_gender.patches:
    Airbnb_users_booked_gender.annotate('{:.1f}'.format(int(k.get_height())), (k.get_x() + 0.16, k.get_height()))


# In[194]:


# Hypothesis testing to determine the relationship between several variables
significance_level = 0.05

# Hypothesis testing about the relationship between country preference and the gender of users
# choose the train_users with true values of country destinations and genders(without unknown elements)
train_users_true = train_users[(train_users['country_destination'] != 'NDF') & (train_users['country_destination'] != 'other') & (train_users['gender'] != 'OTHER') & (train_users['gender'].notnull())]
train_users_true = train_users_true[['id', 'gender', 'country_destination']]

# using chi-square testing
chi_country_gender = train_users_true.pivot_table('id', ['gender'], 'country_destination', aggfunc = 'count').reset_index()
del chi_country_gender.columns.name
chi_country_gender = chi_country_gender.set_index('gender')
chi_country_gender
chi2, p, dof, expected = stats.chi2_contingency(chi_country_gender)

print('p_value:', p)
if p < significance_level:
    print('There exists relationship between country preference and the gender of users!')
else:
    print('There is no relationship between country preference and the gender of users!')
print()

# Hypothesis testing about the relationship between device type and signup method
# choose the train_users with true values of country destinations and genders(without unknown elements)
train_users_signup = train_users[(train_users['signup_method'] != 'google')][['id', 'signup_method', 'signup_app']]
train_users_signup['device'] = train_users_signup['signup_app'].apply(lambda x: 'Computer' if x == 'Web' else 'Mobile')

# using chi_square testing
train_users_signup = train_users_signup.pivot_table('id', ['device'], 'signup_method', aggfunc='count')
train_users_signup.index = ['Computer', 'Mobile']
train_users_signup.columns = ['Basic', 'Facebook']
chi2, p, dof, expected = stats.chi2_contingency(train_users_signup, correction = False)

print('p_value:', p)
if p < significance_level:
    print('There exists relationship between device type and signup method!')
else:
    print('There is no relationship between device type and signup method!')


# In[219]:


# age visualization  

# assign all age values > 100 to NaN, these NaN values will be replaced with real ages below
train_users['age'][train_users['age'] > 100] = np.NaN
test_users['age'][test_users['age'] > 100] = np.NaN

# get average, std, and number of NaN values in airbnb_df
Airbnb_train_age_average = train_users['age'].mean()
Airbnb_train_age_std = train_users['age'].std()
Airbnb_train_age_NAN_count = train_users['age'].isnull().sum()

# get average, std, and number of NaN values in test_df
Airbnb_test_age_average = test_users['age'].mean()
Airbnb_test_age_std = test_users['age'].std()
Airbnb_test_age_NAN_count = test_users['age'].isnull().sum()

# ratio of NaN
Airbnb_train_age_NAN_ratio = Airbnb_train_age_NAN_count/train_users.shape[0]
Airbnb_test_age_NAN_ratio = Airbnb_test_age_NAN_count/test_users.shape[0]
print('The ratio of NaN in age(train set):', '%.6f%%' % (Airbnb_train_age_NAN_ratio * 100))
print('The ratio of NaN in age(test set):', '%.6f%%' % (Airbnb_test_age_NAN_ratio * 100))

# frequency for age values(in case there was a booking) (train_set)
# NaN has not been filled
train_users_d = train_users[train_users['country_destination'] != 'NDF']
train_users_t = train_users_d[train_users_d['age'] <= 100]
plt.figure(figsize = (12, 8))
Airbnb_users_ages = sns.distplot(train_users_t['age'])
plt.title('Number of Airbnb users of different ages(NaN has not been filled) (train_set)')
plt.xlabel('Age')
plt.ylabel('Kernel density estimation of Airbnb users')
Airbnb_users_ages.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


# frequency for age values(in case there was a booking) (test_set)
# NaN has not been filled
test_users_t = test_users[test_users['age'] <= 100]
plt.figure(figsize = (12, 8))
Airbnb_users_ages = sns.distplot(test_users_t['age'])
plt.title('Number of Airbnb users of different ages(NaN has not been filled) (test_set)')
plt.xlabel('Age')
plt.ylabel('Kernel density estimation of Airbnb users')
Airbnb_users_ages.set_cxticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


# from the plot, the distribution is approximately normal distribution
# generate random numbers using truncated normal distribution
import random
import scipy.stats as stats  
lower_bound_age, upper_bound_age = 18, 70     # age boundary
train_rand = stats.truncnorm((lower_bound_age - Airbnb_train_age_average) / Airbnb_train_age_std, 
                             (upper_bound_age - Airbnb_train_age_average) / Airbnb_train_age_std, 
                             loc = Airbnb_train_age_average, scale = Airbnb_train_age_std)
test_rand = stats.truncnorm((lower_bound_age - Airbnb_test_age_average) / Airbnb_test_age_std, 
                             (upper_bound_age - Airbnb_test_age_average) / Airbnb_test_age_std, 
                             loc = Airbnb_test_age_average, scale = Airbnb_test_age_std)


# fill NaN values in Age column with random values generated
train_users['age'][np.isnan(train_users['age'])] = train_rand.rvs(Airbnb_train_age_NAN_count)
test_users['age'][np.isnan(test_users['age'])] = test_rand.rvs(Airbnb_test_age_NAN_count)

# convert type to integer
train_users['age'] = train_users['age'].astype(int)
test_users['age'] = test_users['age'].astype(int)

# frequency for age values(in case there was a booking)

# Convert 100+ into a bin.
age_gender['age_bucket'] = age_gender['age_bucket'].apply(lambda x: '100-104' if x == '100+' else x)

# Define mean_age feature
age_gender['age_mean'] = age_gender['age_bucket'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1]))/2)


# NaN has been filled
plt.figure(figsize = (12, 8))
Airbnb_users_age = sns.distplot(train_users['age'])
plt.title('Number of Airbnb users of different ages(NaN has been filled)')
plt.xlabel('Age')
plt.ylabel('Number of Airbnb users')


# Number of Airbnb users of different age means(in thousands)
plt.figure(figsize = (20, 8))
sns.barplot(x = 'age_mean', y = 'population_in_thousands', hue = 'gender', data = age_gender, ci = None, palette = 'coolwarm')
plt.title('Number of Airbnb users of different age means')
plt.xlabel('Age')
plt.ylabel('Number of Airbnb users of different age means(in thousands)')

    
# cut age values into ranges 
train_users['age_range'] = pd.cut(train_users['age'], [0, 20, 40, 60, 80, 100])
test_users['age_range'] = pd.cut(test_users['age'], [0, 20, 40, 60, 80, 100])


# frequency of country_destination for every age range
plt.figure(figsize = (12, 8))
Airbnb_users_age_country = sns.countplot(x = 'age_range', hue = 'country_destination', data = train_users[train_users['country_destination'] != 'NDF'], palette = 'deep')
plt.title('Number of Airbnb users of different age ranges in different countries')
plt.xlabel('Age range')
plt.ylabel('Number of Airbnb users')


# frequency of booked vs. no-booking users for every gender
plt.figure(figsize = (12, 8))
Airbnb_users_booked_age = sns.countplot(x = 'age_range', hue = 'Booked_or_not', data = train_users)
plt.title('Number of booked vs. non_booking Airbnb users of disparate age groups')
plt.xlabel('Age range')
plt.ylabel('Number of booked and non_booking Airbnb users')

# show the exact number of each country in the histogram
for k in Airbnb_users_booked_age.patches:
    Airbnb_users_booked_age.annotate('{:.1f}'.format(int(k.get_height())), (k.get_x() + 0.08, k.get_height()))


# Statistics of different age groups, the distribution trend of the country using nuclear density statistical methods
plt.figure(figsize = (12, 8))
Airbnb_users_age_country_ct = sns.FacetGrid(data = train_users[train_users['country_destination'] != 'NDF'], hue = 'country_destination', aspect = 4)   
Airbnb_users_age_country_ct.map(sns.kdeplot, 'age', shade = True)
oldest = train_users['age'].max()
Airbnb_users_age_country_ct.set(xlim = (0, oldest))
Airbnb_users_age_country_ct.add_legend()
plt.title('Statistics of the distribution of different age groups in diverse countries(continuous)')
plt.xlabel('Age')
plt.ylabel('Nuclear density estimation')

# Statistics of different age groups, gender distribution trends using nuclear density statistical methods
plt.figure(figsize = (12, 8))
Airbnb_users_age_gender_ct = sns.FacetGrid(data = train_users[train_users['country_destination'] != 'NDF'], hue = 'gender', aspect = 4)   
Airbnb_users_age_gender_ct.map(sns.kdeplot, 'age', shade = True)
oldest = train_users['age'].max()
Airbnb_users_age_gender_ct.set(xlim = (0, oldest))
Airbnb_users_age_gender_ct.add_legend()
plt.title('Statistics of the distribution of different age groups in disparate genders(continuous)')
plt.xlabel('Age')
plt.ylabel('Nuclear density estimation')


# In[196]:


# language visualization
# count of different language users with different country_destination
plt.figure(figsize = (20, 10))
Airbnb_users_language = sns.countplot(x = 'country_destination', hue = 'language', data = train_users[train_users['country_destination'] != 'NDF'])
plt.title('Number of Airbnb users of disparate country_destination using different language')
plt.xlabel('Country_destination')
plt.ylabel('Number of Airbnb users')

country_popularity = train_users[(train_users['country_destination'] != 'NDF') & (train_users['country_destination'] != 'other')]['country_destination'].value_counts()
language_distance = pd.Series(countries['language_levenshtein_distance'])
language_distance.index = countries['country_destination']
lan_con = pd.concat([country_popularity, language_distance], axis = 1)

# plot the jointplot between country_popularity and language_distance
plt.figure(figsize = (12, 8))
lan_con.columns = ['country_popularity', 'language_distance']
sns.jointplot(x = 'country_popularity', y = 'language_distance', data = lan_con.drop('US'))


# In[197]:


# Signup method visualization
plt.figure(figsize = (12, 8))
signup_method = sns.countplot(x = 'signup_method', data = train_users[train_users['country_destination'] != 'NDF'])
plt.title('Number of Airbnb users through different signup methods')
plt.xlabel('Signup methods')
plt.ylabel('Number of Airbnb users')


# In[198]:


# Affiliate channel visualization
plt.figure(figsize = (12, 8))
sns.countplot(train_users['affiliate_channel'])
plt.title('Number of Airbnb users through different affiliate channels')
plt.xlabel('Affiliate channels')
plt.ylabel('Number of Airbnb users')


# In[220]:


# Affiliate provider visualization
plt.figure(figsize = (20, 8))
sns.countplot(x = 'affiliate_provider', hue = 'country_destination', data = train_users[train_users['country_destination'] != 'NDF'], palette = 'deep')
plt.title('Number of Airbnb users through different affiliate providers')
plt.xlabel('Affiliate providers')
plt.ylabel('Number of Airbnb users')


# In[ ]:


# First affiliate tracked visualization
plt.figure(figsize = (10, 8))
sns.countplot(train_users['first_affiliate_tracked'])
plt.title('Number of Airbnb users tracked first through different methods')
plt.xlabel('First affiliate tracked')
plt.ylabel('Number of Airbnb users')


# In[ ]:


# Sessions visualization
# plot the distribution of secs_elapsed
plt.figure(figsize = (12, 8))
sns.distplot(sessions[(sessions['secs_elapsed'].notnull()) & (sessions['secs_elapsed'] < 5000)]['secs_elapsed'], color = 'steelblue')
plt.title('Distribution of secs_elapsed')
plt.xlabel('Secs_elapsed')
plt.ylabel('Nuclear density estimation')

secs_ratio = len(sessions[sessions['secs_elapsed'] < 1000])/len(sessions[sessions['secs_elapsed'].notnull()])
print('Almost %.4f%% of all sessions are less than 1000 seconds.' % (secs_ratio * 100))
                                                            
# count the number of airbnb users using different devices
plt.figure(figsize = (12, 8))
Sessions_device = sns.countplot(y = 'device_type', data = sessions)
plt.title('Number of Airbnb users using different devices')
plt.xlabel('Number of Airbnb users')
plt.ylabel('Device_type')
                                                            


# # Step2: Data Preprocessing

# In[ ]:


# first_affiliate_tracked

from scipy.stats import mode

# fill NaN values with mode
first_affiliate_tracked_type_count = len(np.unique(train_users['first_affiliate_tracked'].value_counts()))    # len(np.unique(train_users["first_affiliate_tracked"].value_counts()))

train_affiliate_tracked_NAN_count = train_users['first_affiliate_tracked'].isnull().sum()
test_affiliate_tracked_NAN_count = test_users['first_affiliate_tracked'].isnull().sum()

train_affiliate_tracked_NAN_ratio = train_affiliate_tracked_NAN_count/train_users.shape[0]
test_affiliate_tracked_NAN_ratio = test_affiliate_tracked_NAN_count/test_users.shape[0]

print('The ratio of NaN in first_affiliate_tracked(train set):', '%.6f%%' % (train_affiliate_tracked_NAN_ratio * 100))
print('The ratio of NaN in first_affiliate_tracked(test set):', '%.6f%%' % (test_affiliate_tracked_NAN_ratio * 100))

#train_rand = np.random.randint(0, first_affiliate_tracked_type_count, size = train_affiliate_tracked_NAN_count)
#test_rand = np.random.randint(0, first_affiliate_tracked_type_count, size = test_affiliate_tracked_NAN_count)

train_affiliate_tracked_type = train_users['first_affiliate_tracked'].value_counts().index
test_affiliate_tracked_type = test_users['first_affiliate_tracked'].value_counts().index                                         

train_users['first_affiliate_tracked'][train_users['first_affiliate_tracked'].isnull()] = train_users['first_affiliate_tracked'].mode().values
test_users['first_affiliate_tracked'][test_users['first_affiliate_tracked'].isnull()] = test_users['first_affiliate_tracked'].mode().values


# In[221]:


# Remove rare values(noise) to speed up computation
import collections
def Remove_Noise(dataframe, features, threshold):
    # Args:
    # df_frame -- A pandas data frame.
    # column_list -- A list of columns.
    # threshold -- The threshold, below which a value is removed.
    insignificant_population = int(np.floor(threshold * len(dataframe)))
    
    print('Remove the insiginficant value of specific features, please be patient...')
    for i in features:
        
        # record the start time of iteration to calculate the running time
        startTime = time.time()
        count = collections.Counter(dataframe[i])
        insignificant_value = []
        for j in count:
            if count[j] < insignificant_population:
                insignificant_value.append(j)
        for k in insignificant_value:
            dataframe[i].replace(k, 'iv', inplace = True)
        print('Insignificant value in the feature:', i, 'has been successfully removed! Took %fs!' % (time.time() - startTime))
    print()


# In[222]:


# Extract frequency counts from pd_frame

def Feature_Frequency_Count(dataframe, features):
    sessions_count = []
    for i in features:
        for j in set(dataframe[i]):
            
            # record the start time of iteration to calculate the running time
            startTime = time.time()
            
            print('Computing frequency counts: %s = %s, please wait...' % (i, j))
            count = dataframe.groupby(dataframe.index).apply(lambda group, x = i, y = j: np.sum(group[x] == y))
            count.name = '%s_%s' % (i, j)
            
            sessions_count.append(count)
            print('Yeah! Extraction succeeds! Costs %fs!' % (time.time() - startTime))
            print()

    frequency_count = pd.concat(sessions_count, axis = 1)
    return frequency_count


# In[ ]:


# Extracting several freatures from the session data
import scipy.stats as stats
sessions_features = ['action', 'action_type', 'action_detail', 'device_type']

# A parameter to speed-up computation. 
# categorical values that appear less than the threshold will be removed.
threshold = 0.003

# load basic training and testing data
sessions.set_index('user_id', inplace = True)
sessions_feature = sessions[sessions_features]
sessions_feature.fillna('NaN', inplace = True)

sessions_secs_elapsed = sessions['secs_elapsed']
print('The median of secs_elapsed:', sessions_secs_elapsed.median())
print('The mean of secs_elapsed:', sessions_secs_elapsed.mean())
print()

# use median to fill NaN
sessions_secs_elapsed.fillna(sessions_secs_elapsed.median(), inplace = True)
sessions['secs_elapsed'] = sessions_secs_elapsed
sessions[sessions_features] = sessions_feature

# extract features from sessions
Remove_Noise(sessions, sessions_features, threshold)
frequency_count = Feature_Frequency_Count(sessions, sessions_features)

# extract simple distribution statistics from a numerical column
sessions_stats = sessions['secs_elapsed'].groupby(sessions.index).aggregate([np.mean, np.std, np.median, np.sum, stats.skew])
sessions_stats.columns = ['secs_elapsed_mean', 'secs_elapsed_std', 'secs_elapsed_median', 'secs_elapsed_sum', 'secs_elapsed_skew']

# Save new data
sessions_feature_extracted = pd.concat((frequency_count, sessions_stats), axis = 1)
sessions_feature_extracted_dataframe = pd.DataFrame(data = sessions_feature_extracted, columns = sessions_feature_extracted.columns, index = sessions_feature_extracted.index)

# Save results
sessions_feature_extracted_dataframe.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/session_features_count.csv')
print('The shape of session_feature_extracted:', sessions_feature_extracted_dataframe.shape)


# In[223]:


# Apply One-Hot-Encoding to categorical features

def One_hot_encoding(dataframe, features):
    feature_dummy = []
    for i in features:
        dataframe_dummy = pd.get_dummies(dataframe[i], prefix = i)
        feature_dummy.append(dataframe_dummy)
    dataframe_new = pd.concat(feature_dummy + [dataframe], axis = 1)
    dataframe_new.drop(features, inplace = True, axis = 1)
    return dataframe_new


# In[224]:


# Preparing the data for the classifiers
# Data proprocessing

import datetime as dt

label = 'country_destination'
feature_cols = ['gender', 'signup_method', 'language', 
                'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 
                'signup_app', 'first_device_type', 'first_browser', 'age_range']

Date = '%Y-%m-%d'                # Expected format for date.
drop_columns = ['timestamp_first_active', 'date_first_booking', 'year_first_booking', 'month_first_booking', 'date_account_created', 'Booked_or_not']


# categorical values that appear less than the threshold will be removed.
threshold = 0.001
train_users.index = train_users['id']
test_users.index = test_users['id']

# load the label columns
label_cols = train_users['country_destination']
train_users.drop(label, inplace = True, axis = 1)

# connect according to vertical axis
# preprocess the train set and test set together
train_test_users = pd.concat((train_users, test_users), axis = 0)   

# remove the noise
Remove_Noise(train_test_users, feature_cols, threshold)

# apply one-hot encoding
train_test_users_oh = One_hot_encoding(train_test_users, feature_cols)

# extract numeric features from date_account_created
date_account_created = train_test_users_oh['date_account_created']
date_account_created_dt = []
year_account_created_dt = []
month_account_created_dt = []
day_account_created_dt = []
weekday_account_created_dt = []

for i in range(date_account_created.shape[0]):
    date_account_created_dt.append(dt.datetime.strptime(str(np.array(date_account_created)[i]), '%Y-%m-%d'))
    year_account_created_dt.append(date_account_created_dt[i].year)
    month_account_created_dt.append(date_account_created_dt[i].month)
    day_account_created_dt.append(date_account_created_dt[i].day)
    weekday_account_created_dt.append(date_account_created_dt[i].weekday())

train_test_users_oh['year_account_created'] = year_account_created_dt
train_test_users_oh['month_account_created'] = month_account_created_dt
train_test_users_oh['day_account_created'] = day_account_created_dt
train_test_users_oh['weekday_account_created'] = weekday_account_created_dt
    
train_test_users_oh.drop(drop_columns, inplace = True, axis = 1)

# load basic data, add session data, and prepare them for predition
sessions_feature_extracted_dataframe = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/session_features_count.csv')
sessions_feature_extracted_dataframe.index = sessions_feature_extracted_dataframe['user_id']

train_test_users_sessions = pd.concat((train_test_users_oh, sessions_feature_extracted_dataframe), axis = 1)
train_users_processed = train_test_users_sessions.ix[train_users.index]
test_users_processed = train_test_users_sessions.ix[test_users.index]

train_users_processed.drop('id',inplace = True, axis = 1)
test_users_processed.drop('id',inplace = True, axis = 1)

train_users_processed.drop('user_id',inplace = True, axis = 1)
test_users_processed.drop('user_id',inplace = True, axis = 1)

# save training, testing data and labels
assert set(train_users_processed.index) == set(label_cols.index)
train_users_processed.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_processed.csv', header = True)
test_users_processed.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_users_processed.csv', header = True)
label_cols.index = train_users_processed.index
label_cols.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_cols.csv', header = True)


# # Step 3: Feature Engineering

# In[31]:


from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_processed.csv')
label_cols = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_cols.csv')
test_users_processed = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_users_processed.csv')

y = label_cols['country_destination']
X = data
X.drop('id', inplace = True, axis = 1)
X.fillna('-1', inplace = True)
test_users_processed.fillna('-1', inplace = True)
numRows, numColumns = X.shape

# encode each country
encoder = LabelEncoder()
encoder.fit(label_cols['country_destination'])
y_encoded = pd.DataFrame(data = encoder.transform(y), columns = ['country destination'])
y_encoded.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_encoded.csv', header = True)

from sklearn.model_selection import KFold
kf = KFold(n_splits = 10)

print(encoder.classes_)
print('The shape of X is is:\n', X.shape)
print('The shape of label is:\n', y_encoded.shape)


# In[244]:


# Feature importances using Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators = 300, random_state = 0)
X_feature = np.array(X.columns)

forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print('Feature ranking:')

for i in range(X.shape[1]):
    print('%d. Feature %s (Relative Importance: %f)' % (i + 1, X_feature[indices[i]], importances[indices[i]]))

# Plot the feature importances of the forest
plt.figure()
plt.title('Feature relative importances ranking')
plt.bar(range(X.shape[1]), importances[indices], color = 'orange', align = 'center')
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[246]:


# Extract the features whose importance >= 0.001
X_processed = [[0 for k in range(125)] for k in range(numRows)]
X_array = X.values
for i in range(numColumns):
    if importances[indices[i]] >= 0.001:
        for j in range(numRows):
            X_processed[j][i] = X_array[j][indices[i]]

columns = X.columns
columns_processed = [0 for k in range(125)]
for i in range(numColumns):
    if importances[indices[i]] >= 0.001:
        columns_processed[i] = columns[indices[i]]
        
X_extracted = pd.DataFrame(data = X_processed, columns = columns_processed)
X_extracted.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/X_extracted.csv', header = True)
test_extracted[X_extracted.columns] = test_users_processed[X_extracted.columns]
test_extracted.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_extracted.csv', header = True)


# In[278]:


# This part is for the final submission
# Feature importances_drop_book with forests of trees 
from sklearn.ensemble import ExtraTreesClassifier

X_drop_book = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_processed.csv', index_col = 0)
label_cols = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_cols.csv', index_col = 0)
test_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_users_processed.csv', index_col = 0)

#X_drop_book.drop('id', inplace = True, axis = 1)
X_drop_book.fillna('-1', inplace = True)
test_users.fillna('-1', inplace = True)
#label_cols.drop('id', inplace = True, axis = 1)

drop_book = ['Booked_or_not_No booking', 'Booked_or_not_Booking']
X_drop_book.drop(drop_book, inplace = True, axis = 1)
test_users.drop(drop_book, inplace = True, axis = 1)

numRows_drop_book, numColumns_drop_book = X_drop_book.shape

# Build a forest and compute the feature importances_drop_book
forest = ExtraTreesClassifier(n_estimators = 300, random_state=0)
forest.fit(X_drop_book, label_cols)
importances_drop_book = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
indices_drop_book = np.argsort(importances_drop_book)[::-1]

# Print the feature ranking
print('Feature ranking:')

for i in range(X_drop_book.shape[1]):
    print('%d. feature %d (%f)' % (i + 1, indices_drop_book[i], importances_drop_book[indices_drop_book[i]]))

# Plot the feature importances of the forest
plt.figure()
plt.title('Feature importances')
plt.bar(range(X_drop_book.shape[1]), importances_drop_book[indices_drop_book], color = 'r', yerr = std[indices_drop_book], align = 'center')
plt.xticks(range(X_drop_book.shape[1]), indices_drop_book)
plt.xlim([-1, X_drop_book.shape[1]])
plt.show()


# In[6]:


from sklearn.decomposition import PCA   
X_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/X_extracted.csv')

pca = PCA(n_components = 4)  
pca.fit(X_extracted)  
X_new = pca.transform(X_extracted)
print(pca.explained_variance_ratio_)  
X_new_dataframe = pd.DataFrame(X_new)


# In[4]:


train_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_2.csv')
test_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_users.csv')
y_encoded = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_encoded.csv', index_col = 0)
label_cols = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_cols.csv')
X_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/X_extracted.csv', index_col = 0)
train_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_extracted.csv', index_col = 0)
test_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_extracted.csv', index_col = 0)
train_users_processed = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_processed.csv', index_col = 0)
test_users_processed = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_users_processed.csv', index_col = 0)


# In[9]:


# Normalization processing to speed-up calculation
from sklearn import preprocessing
numRows, numColumn = X_extracted.shape
X_scaled = preprocessing.scale(X_extracted)     # normalization processing to improve accuracy 


# # Step 4: Model Selection and Training

# In[5]:


# divide into 10 similar subsets (subset_index_array)
# based on stratified sampling of 12 destinations
# serve the Logistic Regression and the multinominal Naive Bayes
AU_num = 0; AU_indexs = [] 
CA_num = 0; CA_indexs = [] 
DE_num = 0; DE_indexs = [] 
ES_num = 0; ES_indexs = [] 
FR_num = 0; FR_indexs = [] 
GB_num = 0; GB_indexs = [] 
IT_num = 0; IT_indexs = [] 
NDF_num = 0; NDF_indexs = [] 
NL_num = 0; NL_indexs = [] 
PT_num = 0; PT_indexs = [] 
US_num = 0; US_indexs = [] 
other_num = 0; other_indexs = [] 

y_array = y_encoded.values

# count the positive and negative samples 
for i in range(len(y_encoded)):       
    if y_array[i] == 0:
        AU_num += 1
        AU_indexs.append(i)
        continue

for i in range(len(y_encoded)):       
    if y_array[i] == 1:
        CA_num += 1
        CA_indexs.append(i)
        continue
    
for i in range(len(y_encoded)):       
    if y_array[i] == 2:
        DE_num += 1
        DE_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 3:
        ES_num += 1
        ES_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 4:
        FR_num += 1
        FR_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 5:
        GB_num += 1
        GB_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 6:
        IT_num += 1
        IT_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 7:
        NDF_num += 1
        NDF_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 8:
        NL_num += 1
        NL_indexs.append(i)
        continue

for i in range(len(y_encoded)):       
    if y_array[i] == 9:
        PT_num += 1
        PT_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 10:
        US_num += 1
        US_indexs.append(i)
        continue
        
for i in range(len(y_encoded)):       
    if y_array[i] == 11:
        other_num += 1
        other_indexs.append(i)
        continue
        
AU_sample = AU_num // 10         
AU_residual = AU_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

CA_sample = CA_num // 10         
CA_residual = CA_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

DE_sample = DE_num // 10         
DE_residual = DE_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

ES_sample = ES_num // 10         
ES_residual = ES_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

FR_sample = FR_num // 10         
FR_residual = FR_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

GB_sample = GB_num // 10         
GB_residual = GB_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

IT_sample = IT_num // 10         
IT_residual = IT_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

NDF_sample = NDF_num // 10         
NDF_residual = NDF_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

NL_sample = NL_num // 10         
NL_residual = NL_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

PT_sample = PT_num // 10         
PT_residual = PT_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

US_sample = US_num // 10         
US_residual = US_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10

other_sample = other_num // 10         
other_residual = other_num % 10     # pos_residual is the remainder of number of positive samples dividing by 10


# evenly distribute data into 10 groups to ensure that each group has similar number of positive and negative samples
AU_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
CA_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
DE_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
ES_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
FR_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
GB_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
IT_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
NDF_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
NL_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
PT_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
US_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]
other_index_array = [[0 for k in range(numRows // 10)] for k in range(10)]

for i in range(0, 10):
    AU_index_array[i] = AU_indexs[i * AU_sample : ((i + 1) * AU_sample)] 
    CA_index_array[i] = CA_indexs[i * CA_sample : ((i + 1) * CA_sample)] 
    DE_index_array[i] = DE_indexs[i * DE_sample : ((i + 1) * DE_sample)] 
    ES_index_array[i] = ES_indexs[i * ES_sample : ((i + 1) * ES_sample)] 
    FR_index_array[i] = FR_indexs[i * FR_sample : ((i + 1) * FR_sample)] 
    GB_index_array[i] = GB_indexs[i * GB_sample : ((i + 1) * GB_sample)] 
    IT_index_array[i] = IT_indexs[i * IT_sample : ((i + 1) * IT_sample)] 
    NDF_index_array[i] = NDF_indexs[i * NDF_sample : ((i + 1) * NDF_sample)] 
    NL_index_array[i] = NL_indexs[i * NL_sample : ((i + 1) * NL_sample)] 
    PT_index_array[i] = PT_indexs[i * PT_sample : ((i + 1) * PT_sample)] 
    US_index_array[i] = US_indexs[i * US_sample : ((i + 1) * US_sample)] 
    other_index_array[i] = other_indexs[i * other_sample : ((i + 1) * other_sample)] 

# put the residual data (both positive and negative samples) into each group evenly
for i in range(AU_residual):
    AU_index_array[i].append(AU_indexs[i + AU_sample * 10])
    
for i in range(CA_residual):
    CA_index_array[i].append(CA_indexs[i + CA_sample * 10])

for i in range(DE_residual):
    DE_index_array[i].append(DE_indexs[i + DE_sample * 10])

for i in range(ES_residual):
    ES_index_array[i].append(ES_indexs[i + ES_sample * 10])
    
for i in range(FR_residual):
    FR_index_array[i].append(FR_indexs[i + FR_sample * 10])
    
for i in range(GB_residual):
    GB_index_array[i].append(GB_indexs[i + GB_sample * 10])
    
for i in range(IT_residual):
    IT_index_array[i].append(IT_indexs[i + IT_sample * 10])
    
for i in range(NDF_residual):
    NDF_index_array[i].append(NDF_indexs[i + NDF_sample * 10])
    
for i in range(NL_residual):
    NL_index_array[i].append(NL_indexs[i + NL_sample * 10])
    
for i in range(PT_residual):
    PT_index_array[i].append(PT_indexs[i + PT_sample * 10])
    
for i in range(US_residual):
    US_index_array[i].append(US_indexs[i + US_sample * 10])
    
for i in range(other_residual):
    other_index_array[i].append(other_indexs[i + other_sample * 10])

# combine positive index array and negative index array together to create 10 subsets
subset_index_array = AU_index_array
for i in range(0, 10):
    for j in range(len(CA_index_array[i])):
        subset_index_array[i].append(CA_index_array[i][j])
    
    for j in range(len(DE_index_array[i])):
        subset_index_array[i].append(DE_index_array[i][j])
    
    for j in range(len(ES_index_array[i])):
        subset_index_array[i].append(ES_index_array[i][j])
    
    for j in range(len(FR_index_array[i])):
        subset_index_array[i].append(FR_index_array[i][j])
    
    for j in range(len(GB_index_array[i])):
        subset_index_array[i].append(GB_index_array[i][j])
    
    for j in range(len(IT_index_array[i])):
        subset_index_array[i].append(IT_index_array[i][j])
    
    for j in range(len(NDF_index_array[i])):
        subset_index_array[i].append(NDF_index_array[i][j])
    
    for j in range(len(NL_index_array[i])):
        subset_index_array[i].append(NL_index_array[i][j])
    
    for j in range(len(PT_index_array[i])):
        subset_index_array[i].append(PT_index_array[i][j])
    
    for j in range(len(US_index_array[i])):
        subset_index_array[i].append(US_index_array[i][j])
    
    for j in range(len(other_index_array[i])):
        subset_index_array[i].append(other_index_array[i][j])

print(AU_num)
print(CA_num)
print(DE_num)
print(ES_num)
print(FR_num)
print(GB_num)
print(IT_num)
print(NDF_num)
print(NL_num)
print(PT_num)
print(US_num)
print(other_num)

print('Number of each subset:')
for i in range(0, 10):
    print(i + 1, len(subset_index_array[i]))


# # NDCG

# In[12]:


# NDCG (Normalized discounted cumulative gain) 
# For each new user, make 5 predictions on the country of the first booking.
# The ground truth country is marked with relevance = 1, while the rest have relevance = 0.

# DCG(Discounted Cumulative Gain)
import math

def DCG(rels):
    dcg = pow(2, rels[0]) - 1
    i = 2
    for rel in rels[1:]:
        dcg = dcg + (pow(2, rel) - 1) / math.log(i + 1, 2)
        i = i + 1
    return dcg

# Largest possible value of DCG
def IDCG(rels):
    rels.sort()
    rels.reverse()
    return DCG(rels)

# NDCG (Normalized discounted cumulative gain) 
def NDCG(predict_proba, y_test):
    # choose the 5 classes with highest probability   
    ids, countries = ([], [])
    for i in range(y_test.shape[0]):
        idx = y_test.index[i]
        ids += [idx] * 5
        countries += (np.argsort(predict_proba[i])[::-1])[:5].tolist()

    result = pd.DataFrame(np.column_stack((ids, countries)), columns = ['id', 'country'])
    result_array = result.values
    y_array = y_test.values
    NDCG_test = []
    
    for i in range(y_test.shape[0]):
        rel = []
        for j in range(5):
            if y_array[i][0] == result_array[i * 5 + j][1]:
                rel.append(1)
            else:
                rel.append(0)
        DCG_test = DCG(rel)
        IDCG_test = IDCG(rel)
        if(IDCG_test == 0):
            NDCG_test.append(1)
        else:
            NDCG_test.append(DCG_test/IDCG_test)
    
    return NDCG_test       


# In[5]:


from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(y_encoded)


# In[20]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (125,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(12, activation = 'softmax'))


# In[21]:


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[22]:


x_val = X_scaled[:20000]
partial_x_train = X_scaled[20000:]
y_val = one_hot_train_labels[:20000]
partial_y_train = one_hot_train_labels[20000:]


# In[12]:


one_hot_train_labels.shape


# In[33]:


history = model.fit(partial_x_train, partial_y_train, epochs = 2, batch_size = 512, validation_data = (x_val, y_val))


# In[24]:


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[25]:


plt.clf() # clear figure
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[34]:


nn = model.predict(test_extracted)


# In[35]:


ids, countries = ([], [])
for i in range(test_extracted.shape[0]):
    idx = test_extracted.index[i]
    ids += [idx] * 5
    countries += encoder.inverse_transform(np.argsort(nn[i])[::-1])[:5].tolist()
submission = pd.DataFrame(np.column_stack((ids, countries)), columns = ['id', 'country'])
submission.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/final_submission_nn.csv', index = False)


# # Logistic Regression

# In[14]:


# logistic regression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report  
from sklearn.metrics import confusion_matrix

predict_total_log = [0 for k in range(numRows)]
predict_proba_log = [[0 for k in range(12)] for k in range(numRows)]
NDCG_total_log = np.zeros((numRows, 1))

# calibrated logistic regression
classifier = CalibratedClassifierCV(LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'), method = 'isotonic', cv = 5)
startTime_log = time.time()

for iteration in range(0, 10):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[subset_index_array[iteration]]
    y_test = y_encoded.loc[subset_index_array[iteration]]
    numRow_range = np.arange(0, numRows, 1)
    train_index = np.delete(numRow_range, subset_index_array[iteration])
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]

    classifier.fit(X_train, y_train)  
    
    prediction = classifier.predict(X_test)  
    proba = classifier.predict_proba(X_test)
    NDCG_log = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(subset_index_array[iteration])):
        predict_total_log[subset_index_array[iteration][j]] = prediction[j]    
    for j in range(len(subset_index_array[iteration])):
        NDCG_total_log[subset_index_array[iteration][j]] = NDCG_log[j]
    
Confusion_matrix_log = confusion_matrix(y_encoded, predict_total_log)
Confusion_matrix_log_dataframe = pd.DataFrame(data = Confusion_matrix_log, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_log_dataframe)
print()

print('NDCG score:')
print(NDCG_total_log.mean())
print()

print('Accuracy score:')  
accuracy_log = accuracy_score(y_encoded, predict_total_log)
print(accuracy_log)  
print()

print('Precision score:')
precision_log = precision_score(y_encoded, predict_total_log, average = 'weighted')
print(precision_log)
print()

print('Recall score:')
recall_log = recall_score(y_encoded, predict_total_log, average = 'weighted')
print(recall_log)
print()

print('The report of classification using logistic regression: ')
print(classification_report(y_encoded, predict_total_log))  

time_log = time.time() - startTime_log
print('The whole logistic regression takes %fs!' % time_log)


# # Decision Tree

# In[13]:


# Decision Tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)
startTime_dt = time.time()

max_depth_dt = 10
min_samples_leaf_dt = 30
min_samples_split_dt = 6
classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = max_depth_dt, 
                                    min_samples_leaf = min_samples_leaf_dt,
                                    min_samples_split = min_samples_split_dt, random_state = 0)
predict_total_dt = []
NDCG_total_dt = []
iteration = 0
for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
     
    classifier.fit(X_train, y_train)  
    
    prediction = classifier.predict(X_test)  
    proba = classifier.predict_proba(X_test)
    NDCG_dt = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_dt.append(prediction[j])
    for j in range(len(NDCG_dt)):
        NDCG_total_dt.append(NDCG_dt[j])
    iteration += 1
    
Confusion_matrix_dt = confusion_matrix(y_encoded, predict_total_dt)
Confusion_matrix_dt_dataframe = pd.DataFrame(data = Confusion_matrix_dt, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_dt_dataframe)
print()

print('NDGG score:')
NDCG_total_dt = np.array(NDCG_total_dt)
print(NDCG_total_dt.mean())
print()

print('Accuracy score:')  
accuracy_dt = accuracy_score(y_encoded, predict_total_dt)
print(accuracy_dt)  
print()

print('Precision score:')
precision_dt = precision_score(y_encoded, predict_total_dt, average = 'weighted')
print(precision_dt)
print()

print('Recall score:')
recall_dt = recall_score(y_encoded, predict_total_dt, average = 'weighted')
print(recall_dt)
print()

print('The report of classification using decision tree: ')
print(classification_report(y_encoded, predict_total_dt))  

time_dt = time.time() - startTime_dt
print('The whole decision tree takes %fs!' % time_dt)


# # Naive Bayes

# In[9]:


# Multinominal Naive Bayes
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report  
from sklearn.metrics import confusion_matrix

startTime_nb = time.time()
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X_extracted)

# calibrated Naive Bayes
classifier = CalibratedClassifierCV(MultinomialNB(random_state = 0), method = 'isotonic', cv = 5)

predict_total_nb = [0 for k in range(numRows)]
predict_proba_nb = [[0 for k in range(12)] for k in range(numRows)]
NDCG_total_nb = np.zeros((numRows, 1))
for iteration in range(0, 10):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_minmax[subset_index_array[iteration]]
    y_test = y_encoded.loc[subset_index_array[iteration]]
    numRow_range = np.arange(0, numRows, 1)
    train_index = np.delete(numRow_range, subset_index_array[iteration])
    X_train = X_minmax[train_index]
    y_train = y_encoded.loc[train_index]
    
    y_train_processed = []
    for i in range(y_train.shape[0]):
        y_train_processed.append(y_train.values[i][0])
    classifier.fit(X_train, y_train_processed)  
    
    prediction = classifier.predict(X_test) 
    proba = classifier.predict_proba(X_test)
    NDCG_nb = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(subset_index_array[iteration])):
        predict_total_nb[subset_index_array[iteration][j]] = prediction[j]
    
    for j in range(len(subset_index_array[iteration])):
        NDCG_total_nb[subset_index_array[iteration][j]] = NDCG_nb[j]

    
Confusion_matrix_nb = confusion_matrix(y_encoded, predict_total_nb)
Confusion_matrix_nb_dataframe = pd.DataFrame(data = Confusion_matrix_nb, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_nb_dataframe)
print()

print('NDGG score:')
print(NDCG_total_nb.mean())
print()

print('Accuracy score:')  
accuracy_nb = accuracy_score(y_encoded, predict_total_nb)
print(accuracy_nb)  
print()

print('Precision score:')
precision_nb = precision_score(y_encoded, predict_total_nb, average = 'weighted')
print(precision_nb)
print()

print('Recall score:')
recall_nb = recall_score(y_encoded, predict_total_nb, average = 'weighted')
print(recall_nb)
print()

print('The report of classification using multinominal naive bayes: ')
print(classification_report(y_encoded, predict_total_nb))  

time_nb = time.time() - startTime_nb
print('The whole naive bayes takes %fs!' % time_nb)


# # Random Forest

# In[ ]:


# Parameters tuning for random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics

# determine the optimal parameters using 10-fold cross-validation
# n_estimators
n_estimators_test = {'n_estimators': [100, 150, 200]}
gsearch_n_estimators = GridSearchCV(estimator = RandomForestClassifier(), param_grid = n_estimators_test, cv = 10)
gsearch_n_estimators.fit(X_train, y_train)
n_estimators_optimal = gsearch_n_estimators.best_params_['n_estimators']
print('The appropriate n_estimators is:', n_estimators_optimal)

# max_depth, min_samples_split, min_sample_leaf
depth_split_leaf_test = {'max_depth': [20, 25, 30], 
                         'min_samples_split': [6, 8, 12],
                         'min_samples_leaf':[20, 25, 30]}
gsearch_depth_split_leaf = GridSearchCV(estimator = RandomForestClassifier(n_estimators = n_estimators_optimal), 
                                        param_grid = depth_split_leaf_test, scoring = 'recall', cv = 10)

gsearch_depth_split_leaf.fit(X_train, y_train)
max_depth_optimal = gsearch_depth_split_leaf.best_params_['max_depth']
min_samples_split_optimal = gsearch_depth_split_leaf.best_params_['min_samples_split']
min_samples_leaf_optimal = gsearch_depth_split_leaf.best_params_['min_samples_leaf']
print('The appropriate max_depth is:', max_depth_optimal)
print('The appropriate min_samples_split is:', min_samples_split_optimal)
print('The appropriate min_samples_leaf is:', min_samples_leaf_optimal)

# n_jobs
n_jobs_test = {'n_jobs': [20, 30]}
gsearch_n_jobs =  GridSearchCV(estimator = RandomForestClassifier(n_estimators = n_estimators_optimal, 
                                                                  max_depth = max_depth_optimal, 
                                                                  min_samples_split =  min_samples_split_optimal, 
                                                                  min_samples_leaf = min_samples_leaf_optimal), 
                               param_grid = n_jobs_test, scoring = 'recall', cv = 10)
gsearch_n_jobs.fit(X_train, y_train)
n_jobs_optimal = gsearch_n_jobs.best_params_['n_jobs']
print('The appropriate n_jobs is:', n_jobs_optimal)


# In[15]:


# Random Forest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)
startTime_rf = time.time()

n_estimators_rf = 150
n_jobs_rf = 20
max_depth_rf = 20
min_samples_leaf_rf = 30
min_samples_split_rf = 20
classifier = RandomForestClassifier(n_estimators = n_estimators_rf, n_jobs = n_jobs_rf,
                                    max_depth = max_depth_rf, min_samples_leaf = min_samples_leaf_rf,
                                    min_samples_split = min_samples_split_rf, random_state = 0,
                                    bootstrap = True)
predict_total_rf = []
NDCG_total_rf = []
iteration = 0

for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
     
    classifier.fit(X_train, y_train)  
    prediction = classifier.predict(X_test)  
    proba = classifier.predict_proba(X_test)
    NDCG_rf = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_rf.append(prediction[j]) 
    for j in range(len(NDCG_rf)):
        NDCG_total_rf.append(NDCG_rf[j])
    iteration += 1
    
Confusion_matrix_rf = confusion_matrix(y_encoded, predict_total_rf)
Confusion_matrix_rf_dataframe = pd.DataFrame(data = Confusion_matrix_rf, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_rf_dataframe)
print()

print('NDGG score:')
NDCG_total_rf = np.array(NDCG_total_rf)
print(NDCG_total_rf.mean())
print()

print('Accuracy score:')  
accuracy_rf = accuracy_score(y_encoded, predict_total_rf)
print(accuracy_rf)  
print()

print('Precision score:')
precision_rf = precision_score(y_encoded, predict_total_rf, average = 'weighted')
print(precision_rf)
print()

print('Recall score:')
recall_rf = recall_score(y_encoded, predict_total_rf, average = 'weighted')
print(recall_rf)
print()

print('The report of classification using random forest: ')
print(classification_report(y_encoded, predict_total_rf))  

time_rf = time.time() - startTime_rf
print('The whole random forest takes %fs!' % time_rf)


# # Extra Trees Classifier

# In[16]:


# Extra Trees Classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)
startTime_etc = time.time()

n_estimators_etc = 150
n_jobs_etc = 20
max_depth_etc = 20
min_samples_leaf_etc = 30
min_samples_split_etc = 20
classifier = ExtraTreesClassifier(n_estimators = n_estimators_etc, n_jobs = n_jobs_etc,
                                    max_depth = max_depth_etc, min_samples_leaf = min_samples_leaf_etc,
                                    min_samples_split = min_samples_split_etc, random_state = 0,
                                    bootstrap = True)
predict_total_etc = []
NDCG_total_etc = []
iteration = 0

for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
     
    classifier.fit(X_train, y_train)  
    prediction = classifier.predict(X_test)  
    proba = classifier.predict_proba(X_test)
    NDCG_etc = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_etc.append(prediction[j]) 
    for j in range(len(NDCG_etc)):
        NDCG_total_etc.append(NDCG_etc[j])
    iteration += 1
    
Confusion_matrix_etc = confusion_matrix(y_encoded, predict_total_etc)
Confusion_matrix_etc_dataframe = pd.DataFrame(data = Confusion_matrix_etc, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_etc_dataframe)
print()

print('NDGG score:')
NDCG_total_etc = np.array(NDCG_total_etc)
print(NDCG_total_etc.mean())
print()

print('Accuracy score:')  
accuracy_etc = accuracy_score(y_encoded, predict_total_etc)
print(accuracy_etc)  
print()

print('Precision score:')
precision_etc = precision_score(y_encoded, predict_total_etc, average = 'weighted')
print(precision_etc)
print()

print('Recall score:')
recall_etc = recall_score(y_encoded, predict_total_etc, average = 'weighted')
print(recall_etc)
print()

print('The report of classification using extra tree classifier: ')
print(classification_report(y_encoded, predict_total_etc))  

time_etc = time.time() - startTime_etc
print('The whole extra tree classifier takes %fs!' % time_etc)


# # Xgboost

# In[ ]:


# Compute the optimal parameters
from xgboost.sklearn import XGBClassifier
tuned_params = [{'objective': ['multi:softprob'], 'learning_rate': [0.08, 0.1, 0.16], 
                 'n_estimators': [80, 100, 120], 'max_depth':[4, 7, 12]}]
begin_t = time.time()
clf = GridSearchCV(xgb.XGBClassifier(seed = 7), tuned_params, scoring = 'recall')
y_encoded_processed = []
for i in range(y_encoded.shape[0]):
    y_encoded_processed.append(y_encoded.values[i][0])
    
clf.fit(X_scaled, y_encoded_processed)
end_t = time.time()
print('Train time: ', round(end_t - begin_t, 3), 's')
print('Current best parameters of xgboost: ',clf.best_params_)
print(clf.best_estimator_)


# In[ ]:


# Xgboost
from sklearn.calibration import CalibratedClassifierCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)
startTime_xgb = time.time()

# Predictions using xgboost
classifier = XGBClassifier(objective = 'multi:softprob', learning_rate = 0.08, 
                    max_depth = 7, n_estimators = 100, 
                    subsample = 0.6, colsample_bytree = 0.8)

predict_total_xgb = []
NDCG_total_xgb = []
iteration = 0
for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
     
    classifier.fit(X_train, y_train)  
    
    prediction = classifier.predict(X_test)  
    proba = classifier.predict_proba(X_test)
    NDCG_xgb = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_xgb.append(prediction[j])
    for j in range(len(NDCG_xgb)):
        NDCG_total_xgb.append(NDCG_xgb[j])
    iteration += 1
    
Confusion_matrix_xgb = confusion_matrix(y_encoded, predict_total_xgb)
Confusion_matrix_xgb_dataframe = pd.DataFrame(data = Confusion_matrix_xgb, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_xgb_dataframe)
print()

print('NDGG score:')
NDCG_total_xgb = np.array(NDCG_total_xgb)
print(NDCG_total_xgb.mean())
print()

print('Accuracy score:')  
accuracy_xgb = accuracy_score(y_encoded, predict_total_xgb)
print(accuracy_xgb)  
print()

print('Precision score:')
precision_xgb = precision_score(y_encoded, predict_total_xgb, average = 'weighted')
print(precision_xgb)
print()

print('Recall score:')
recall_xgb = recall_score(y_encoded, predict_total_xgb, average = 'weighted')
print(recall_xgb)
print()

print('The report of classification using xgboost: ')
print(classification_report(y_encoded, predict_total_xgb))  

time_xgb = time.time() - startTime_xgb
print('The whole xgboost takes %fs!' % time_xgb)


# # AdaBoost

# In[18]:


# AdaBoost Classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)
startTime_ab = time.time()

# Predictions using Adaoosting
n_estimators_ab = 100
learning_rate_ab = 0.1
classifier = CalibratedClassifierCV(AdaBoostClassifier(n_estimators = n_estimators_ab, learning_rate = learning_rate_ab, random_state = 0), method = 'isotonic', cv = 5)

predict_total_ab = []
NDCG_total_ab = []
iteration = 0

for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
    
    classifier.fit(X_train, y_train)  
    
    prediction = classifier.predict(X_test) 
    proba = classifier.predict_proba(X_test)
    NDCG_ab = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_ab.append(prediction[j])    
    for j in range(len(NDCG_ab)):
        NDCG_total_ab.append(NDCG_ab[j])
        
    iteration += 1

Confusion_matrix_ab = confusion_matrix(y_encoded, predict_total_ab)
Confusion_matrix_ab_dataframe = pd.DataFrame(data = Confusion_matrix_ab, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'ab(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'ab(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_ab_dataframe)
print()

print('NDGG score:')
NDCG_total_ab = np.array(NDCG_total_ab)
print(NDCG_total_ab.mean())
print()

print('Accuracy score:')  
accuracy_ab = accuracy_score(y_encoded, predict_total_ab)
print(accuracy_ab)  
print()

print('Precision score:')
precision_ab = precision_score(y_encoded, predict_total_ab, average = 'weighted')
print(precision_ab)
print()

print('Recall score:')
recall_ab = recall_score(y_encoded, predict_total_ab, average = 'weighted')
print(recall_ab)
print()

print('The report of classification using adaboost: ')
print(classification_report(y_encoded, predict_total_ab))  

time_ab = time.time() - startTime_ab
print('The whole adaboost takes %fs!' % time_ab)


# # Light Gradient Boosting Machine

# In[ ]:


# Compute the optimal parameters
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
tuned_params = [{'learning_rate': [0.1, 0.2, 0.3],
                 'n_estimators': [100, 150, 200], 'max_depth':[4, 6, 8, 9]}]

begin_t = time.time()
model = lgb.LGBMClassifier(objective = 'multiclass', seed = 42)  
clf = GridSearchCV(estimator = model, param_grid = tuned_params, cv = 3)
y_encoded_processed = []
for i in range(y_encoded.shape[0]):
        y_encoded_processed.append(y_encoded.values[i][0])

clf.fit(X_scaled, y_encoded_processed)
end_t = time.time()

print('Training time: ', round(end_t - begin_t, 3), 's')
print('Current optimal parameters of lgb: ',clf.best_params_)
print(clf.best_estimator_)


# In[57]:


# Light Gradient Boosting Machine
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)
startTime_lgb = time.time()

# Predictions using ligntgbm
max_depth_lgb = 4
learning_rate_lgb = 0.1

# LightGBM
classifier = lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.1,
                                max_bin = 255, max_depth = 4, min_child_samples = 10, min_child_weight = 5,
                                min_split_gain = 0.0, n_estimators = 200, objective = 'multiclass',
                                random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42, 
                                silent = True, subsample = 1.0,
                                subsample_for_bin = 50000, subsample_freq = 1)
    
params = {'objective': 'multiclass', 'num_class': 12, 'learning_rate': 0.08, 'max_depth': 16, 'seed': 7, 'n_estimators': 100}
kf = KFold(n_splits = 10)

predict_total_lgb = []
NDCG_total_lgb = []
iteration = 0
for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
    
    y_train_processed = []
    for i in range(y_train.shape[0]):
        y_train_processed.append(y_train.values[i][0])
    
    classifier.fit(X_train, y_train_processed)
    prediction = classifier.predict(X_test) 
    proba = classifier.predict_proba(X_test)
    NDCG_lgb = NDCG(proba, y_test)
       
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_lgb.append(prediction[j]) 
    for j in range(len(NDCG_lgb)):
        NDCG_total_lgb.append(NDCG_lgb[j])
    iteration += 1
    
    
Confusion_matrix_lgb = confusion_matrix(y_encoded, predict_total_lgb)
Confusion_matrix_lgb_dataframe = pd.DataFrame(data = Confusion_matrix_lgb, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_lgb_dataframe)
print()

print('NDGG score:')
NDCG_total_lgb = np.array(NDCG_total_lgb)
print(NDCG_total_lgb.mean())
print()

print('Accuracy score:')  
accuracy_lgb = accuracy_score(y_encoded, predict_total_lgb)
print(accuracy_lgb)  
print()

print('Precision score:')
precision_lgb = precision_score(y_encoded, predict_total_lgb, average = 'weighted')
print(precision_lgb)
print()

print('Recall score:')
recall_lgb = recall_score(y_encoded, predict_total_lgb, average = 'weighted')
print(recall_lgb)
print()

print('The report of classification using lightgbm: ')
print(classification_report(y_encoded, predict_total_lgb))  

time_lgb = time.time() - startTime_lgb
print('The whole lightgbm takes %fs!' % time_lgb)


# # Step 5: Model Ensemble

# # Model Performance Comparison

# In[97]:


# plot the ROC curve
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn import datasets
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

def ROC_curve_AUC(cf, X, y):
    # Binarize the output
    y_bin = label_binarize(y, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    class_num = y_bin.shape[1]
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size = 0.1, random_state = 0)
    
    if cf == 'Logistic Regression' :
        classifier = OneVsRestClassifier(CalibratedClassifierCV(LogisticRegression(random_state = 0), method = 'isotonic'))
        classifier.fit(X_train, y_train)  
        proba =  classifier.decision_function(X_test)
        y_bin_score = proba
        
    if cf == 'Random Forest' :
        n_estimators_rf = 150
        n_jobs_rf = 20
        max_depth_rf = 20
        min_samples_leaf_rf = 30
        min_samples_split_rf = 20
        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators = n_estimators_rf, n_jobs = n_jobs_rf,
                                            max_depth = max_depth_rf, min_samples_leaf = min_samples_leaf_rf,
                                            min_samples_split = min_samples_split_rf, random_state = 0,
                                            bootstrap = True))
        classifier.fit(X_train, y_train)  
        y_bin_score =  classifier.predict_proba(X_test)
    
    if cf == 'Decision Tree' :
        max_depth_dt = 10
        min_samples_leaf_dt = 30
        min_samples_split_dt = 6
        classifier = OneVsRestClassifier(DecisionTreeClassifier(criterion = 'gini', max_depth = max_depth_dt, 
                                            min_samples_leaf = min_samples_leaf_dt,
                                            min_samples_split = min_samples_split_dt))
        classifier.fit(X_train, y_train)  
        y_bin_score =  classifier.predict_proba(X_test)
        
    if cf == 'Extra Trees' :
        n_estimators_etc = 150
        n_jobs_etc = 20
        max_depth_etc = 20
        min_samples_leaf_etc = 30
        min_samples_split_etc = 20
        classifier = OneVsRestClassifier(ExtraTreesClassifier(n_estimators = n_estimators_etc, n_jobs = n_jobs_etc,
                                          max_depth = max_depth_etc, min_samples_leaf = min_samples_leaf_etc,
                                          min_samples_split = min_samples_split_etc, random_state = 0,
                                          bootstrap = True))
        y_bin_score =  classifier.predict_proba(X_test)
        
    if cf == 'Naive Bayes' :
        classifier = OneVsRestClassifier(GaussianNB())
        classifier.fit(X_train, y_train)  
        y_bin_score =  classifier.predict_proba(X_test)
        
    if cf == 'Xgboost' :
        classifier = OneVsRestClassifier(XGBClassifier(learning_rate = 0.1, max_depth = 7, n_estimators = 80, 
                                   subsample = 0.6, colsample_bytree = 0.8, seed = 0))
        classifier.fit(X_train, y_train)  
        y_bin_score =  classifier.predict_proba(X_test)
    
    if cf == 'AdaBoost':
        n_estimators_ab = 100
        learning_rate_ab = 0.1
        classifier = OneVsRestClassifier(AdaBoostClassifier(n_estimators = n_estimators_ab, 
                                                            learning_rate = learning_rate_ab, random_state = 0))
        classifier.fit(X_train, y_train)  
        y_bin_score =  classifier.decision_function(X_test)
        
    if cf == 'LightGBM':
        classifier = OneVsRestClassifier(lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.1,
                                        max_bin = 255, max_depth = 4, min_child_samples = 10, min_child_weight = 5,
                                        min_split_gain = 0.0, n_estimators = 200, objective = 'multiclass',
                                        random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42, 
                                        silent = True, subsample = 1.0,
                                        subsample_for_bin = 50000, subsample_freq = 1))
        classifier.fit(X_train, y_train)  
        y_bin_score = classifier.predict_proba(X_test)
        
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(class_num):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_bin_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= class_num
    
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    
    # Plot all ROC curves
    plt.figure()
    
    plt.plot(fpr['macro'], tpr['macro'], label = 'macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),
             color = 'navy', linestyle = ':', linewidth = 4)
    
    color_range = ['aqua', 'darkorange', 'cornflowerblue', 'tomato', 'violet', 'indigo', 
                   'lime', 'orange', 'olive', 'saddlebrown', 'steelblue', 'pink']
    
    for i in range(class_num):
        plt.plot(fpr[i], tpr[i], lw = 2, color = color_range[i], label = 'ROC curve of class {0} (AUC = {1:0.2f})'''.format(encoder.classes_[i], roc_auc[i]))
    
    
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)  # plot the diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of multi-classes and macro-average ROC curve')
    plt.legend(loc = "lower right")
    plt.legend(bbox_to_anchor = (1.0, 1.0))
    plt.show()


# In[ ]:


# plot the roc curve of 12 classes with heterogenerous classifiers
''' 
    Classifiers are:
       1. 'Logistic Regression'
       2. 'Decision Tree'
       3. 'Naive Bayes'
       4. 'Random Forest'
       5. 'Extra Trees'
       6. 'Xgboost'
       7. 'AdaBoost'
       8. 'LightGBM'
'''

# Logistic regression          
print('ROC curve and AUC using logistic regression:')
ROC_curve_AUC('Logistic Regression', X_scaled, y_encoded)

# Decision Tree
print('ROC curve and AUC using decision tree:')
ROC_curve_AUC('Decision Tree', X_scaled, y_encoded)

# Naive Bayes
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X_extracted)
print('ROC curve and AUC using naive bayes:')
ROC_curve_AUC('Naive Bayes', X_minmax, y_encoded)

# Random Forest
print('ROC curve and AUC using random forest:')
ROC_curve_AUC('Random Forest', X_scaled, y_encoded)

# Extra Trees Classifier
print('ROC curve and AUC using extra trees:')
ROC_curve_AUC('Extra Trees', X_scaled, y_encoded)

# Xgboost
print('ROC curve and AUC using xgboost:')
ROC_curve_AUC('Xgboost', X_scaled, y_encoded)

# AdaBoost
print('ROC curve and AUC using adaboost:')
ROC_curve_AUC('AdaBoost', X_scaled, y_encoded)

# LightGBM      
print('ROC curve and AUC using light gradient boosting machine:')
ROC_curve_AUC('LightGBM', X_scaled, y_encoded)


# In[23]:


# plot the histogram of NDCG, accuracy, precision, recall, running time of each classifier

# NDCG
x_model = np.arange(8)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM']
NDCG_model = [NDCG_total_log.mean(), NDCG_total_dt.mean(), NDCG_total_nb.mean(), NDCG_total_rf.mean(), NDCG_total_etc.mean(), NDCG_total_xgb.mean(), NDCG_total_ab.mean(), NDCG_total_lgb.mean()]
barlist = plt.bar(x_model, NDCG_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of NDCG
NDCG_max = max(NDCG_model)
idx = NDCG_model.index(NDCG_max)
barlist[idx].set_color('r')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('NDCG of each model')
plt.title('Model performance(NDCG)')
plt.show()
print('Model {0} has the highest NDCG score: {1}'.format(model[idx], NDCG_max))


# In[24]:


# plot the histogram of NDCG, accuracy, precision, recall, running time of each classifier

# accuracy
x_model = np.arange(8)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM']
accuracy_model = [accuracy_log, accuracy_dt, accuracy_nb, accuracy_rf, accuracy_etc, accuracy_xgb, accuracy_ab, accuracy_lgb]
barlist = plt.bar(x_model, accuracy_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of accuracy
accuracy_max = max(accuracy_model)
idx = accuracy_model.index(accuracy_max)
barlist[idx].set_color('g')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('Accuracy of each model')
plt.title('Model performance(accuracy)')
plt.show()
print('Model {0} has the highest accuracy score: {1}'.format(model[idx], accuracy_max))


# In[25]:


# plot the histogram of NDCG, accuracy, precision, recall, running time of each classifier

# precision
x_model = np.arange(8)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM']
precision_model = [precision_log, precision_dt, precision_nb, precision_rf, precision_etc, precision_xgb, precision_ab, precision_lgb]
barlist = plt.bar(x_model, precision_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of precision
precision_max = max(precision_model)
idx = precision_model.index(precision_max)
barlist[idx].set_color('orange')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('Precision of each model')
plt.title('Model performance(precision)')
plt.show()
print('Model {0} has the highest precision score: {1}'.format(model[idx], precision_max))


# In[26]:


# plot the histogram of NDCG, accuracy, recall, recall, running time of each classifier

# recall
x_model = np.arange(8)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM']
recall_model = [recall_log, recall_dt, recall_nb, recall_rf, recall_etc, recall_xgb, recall_ab, recall_lgb]
barlist = plt.bar(x_model, recall_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of recall
recall_max = max(recall_model)
idx = recall_model.index(recall_max)
barlist[idx].set_color('indigo')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('Recall of each model')
plt.title('Model performance(recall)')
plt.show()
print('Model {0} has the highest recall score: {1}'.format(model[idx], recall_max))


# In[27]:


# plot the histogram of NDCG, accuracy, precision, recall, running time of each classifier

# time
x_model = np.arange(8)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM']
time_model = [time_log, time_dt, time_nb, time_rf, time_etc, time_xgb, time_ab, time_lgb]
barlist = plt.bar(x_model, time_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of time
time_min = min(time_model)
idx = time_model.index(time_min)
barlist[idx].set_color('y')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('Running time of each model')
plt.title('Model performance(running time)')
plt.show()
print('Model {0} has the shortest running time: {1}s'.format(model[idx], time_min))


# # Stacking

# In[29]:


from mlxtend.classifier import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# stacking strategy using xgboost, lighgbm and random forest

# meta classifier is the decision tree
max_depth_dt = 10
min_samples_leaf_dt = 30
min_samples_split_dt = 6
meta_clf = CalibratedClassifierCV(DecisionTreeClassifier(criterion = 'gini', max_depth = max_depth_dt, 
                                  min_samples_leaf = min_samples_leaf_dt,
                                  min_samples_split = min_samples_split_dt), method = 'isotonic', cv = 5)

# Xgboost
clf1 = XGBClassifier(objective = 'multi:softprob', learning_rate = 0.08, 
                     max_depth = 10, n_estimators = 100, 
                     subsample = 0.6, colsample_bytree = 0.8, seed = 0)

# LightGBM
clf2 = lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.08,
                          max_bin = 255, max_depth = 16, min_child_samples = 10, min_child_weight = 5,
                          min_split_gain = 0.0, n_estimators = 100, objective = 'multiclass',
                          random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42, 
                          silent = True, subsample = 1.0,
                          subsample_for_bin = 50000, subsample_freq = 1)

# Random Forest
n_estimators_rf = 150
n_jobs_rf = 20
max_depth_rf = 20
min_samples_leaf_rf = 30
min_samples_split_rf = 20
clf3 = RandomForestClassifier(n_estimators = n_estimators_rf, n_jobs = n_jobs_rf,
                              max_depth = max_depth_rf, min_samples_leaf = min_samples_leaf_rf,
                              min_samples_split = min_samples_split_rf, random_state = 0,
                              bootstrap = True)

stacking_clf = StackingClassifier(classifiers = [clf1, clf2, clf3], meta_classifier = meta_clf)
kf = KFold(n_splits = 10)
startTime_stacking = time.time()

predict_total_stacking = []
NDCG_total_stacking = []
iteration = 0
for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
    
    y_train_processed = []
    for i in range(y_train.shape[0]):
        y_train_processed.append(y_train.values[i][0])
    
    stacking_clf.fit(X_train, y_train_processed)
    prediction = stacking_clf.predict(X_test) 
    proba = stacking_clf.predict_proba(X_test)
    NDCG_stacking = NDCG(proba, y_test)
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_stacking.append(prediction[j]) 
    for j in range(len(NDCG_stacking)):
        NDCG_total_stacking.append(NDCG_stacking[j])
        
    iteration += 1
    
    
Confusion_matrix_stacking = confusion_matrix(y_encoded, predict_total_stacking)
Confusion_matrix_stacking_dataframe = pd.DataFrame(data = Confusion_matrix_stacking, 
                                                   index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'stacking(true)',
                                                            'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                                   columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'stacking(prediction)',
                                                              'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_stacking_dataframe)
print()

print('NDGG score:')
NDCG_total_stacking = np.array(NDCG_total_stacking)
print(NDCG_total_stacking.mean())
print()

print('Accuracy score:')  
accuracy_stacking = accuracy_score(y_encoded, predict_total_stacking)
print(accuracy_stacking)  
print()

print('Precision score:')
precision_stacking = precision_score(y_encoded, predict_total_stacking, average = 'weighted')
print(precision_stacking)
print()

print('Recall score:')
recall_stacking = recall_score(y_encoded, predict_total_stacking, average = 'weighted')
print(recall_stacking)
print()

print('The report of classification using stacking classifier: ')
print(classification_report(y_encoded, predict_total_stacking))  

time_stacking = time.time() - startTime_stacking
print('The whole stacking classifier takes %fs!' % time_stacking)


# # Voting

# In[ ]:


# Voting classifier including Xgboost, Random Forest, LightGBM
# weights are: Xgboost 5, Random Forest 1, LightGBM 2
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression 
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb

from sklearn.metrics import accuracy_score, classification_report  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# Xgboost
clf1 = CalibratedClassifierCV(XGBClassifier(objective = 'multi:softprob', learning_rate = 0.08, 
                     max_depth = 19, n_estimators = 100, 
                     subsample = 0.6, colsample_bytree = 0.8, seed = 0), method = 'isotonic')

# Random Forest
n_estimators_rf = 150
n_jobs_rf = 20
max_depth_rf = 20
min_samples_leaf_rf = 30
min_samples_split_rf = 20
clf2 = CalibratedClassifierCV(RandomForestClassifier(n_estimators = n_estimators_rf, n_jobs = n_jobs_rf,
                              max_depth = max_depth_rf, min_samples_leaf = min_samples_leaf_rf,
                              min_samples_split = min_samples_split_rf, random_state = 0,
                              bootstrap = True), method = 'isotonic')

# LightGBM
clf3 = CalibratedClassifierCV(lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.08,
                                max_bin = 255, max_depth = 16, min_child_samples = 10, min_child_weight = 5,
                                min_split_gain = 0.0, n_estimators = 200, objective = 'multiclass',
                                random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42, 
                                silent = True, subsample = 1.0,
                                subsample_for_bin = 50000, subsample_freq = 1), method = 'isotonic')
    

# Voting classifier
voting_clf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('lgb', clf3)], 
                              voting = 'soft', weights = [5, 1, 2])

kf = KFold(n_splits = 10)
startTime_vote = time.time()

predict_total_vote = []
NDCG_total_vote = []

iteration = 0
for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
    
    y_train_processed = []
    for i in range(y_train.shape[0]):
        y_train_processed.append(y_train.values[i][0])
        
    voting_clf.fit(X_train, y_train_processed)
    prediction = voting_clf.predict(X_test) 
    proba = voting_clf.predict_proba(X_test)
    NDCG_vote = NDCG(proba, y_test)
        
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_vote.append(prediction[j]) 
    for j in range(len(NDCG_vote)):
        NDCG_total_vote.append(NDCG_vote[j])
    iteration += 1
    
    
Confusion_matrix_vote = confusion_matrix(y_encoded, predict_total_vote)
Confusion_matrix_vote_dataframe = pd.DataFrame(data = Confusion_matrix_vote, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'vote(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'vote(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_vote_dataframe)
print()

print('NDGG score:')
NDCG_total_vote = np.array(NDCG_total_vote)
print(NDCG_total_vote.mean())
print()

print('Accuracy score:')  
accuracy_vote = accuracy_score(y_encoded, predict_total_vote)
print(accuracy_vote)  
print()

print('Precision score:')
precision_vote = precision_score(y_encoded, predict_total_vote, average = 'weighted')
print(precision_vote)
print()

print('Recall score:')
recall_vote = recall_score(y_encoded, predict_total_vote, average = 'weighted')
print(recall_vote)
print()

print('The report of classification using voting classifier: ')
print(classification_report(y_encoded, predict_total_vote))  

time_vote = time.time() - startTime_vote
print('The whole voting classifier takes %fs!' % time_vote)


# # Comparison Between Former Classifiers and Ensemble Models

# In[61]:


from sklearn.metrics import log_loss
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier

import matplotlib.pylab as plt

# For each value in classes, a dataset with that number of classes will be created. 
classes = range(3, 15)


ll_rf = []   # to store logloss of random forest
ll_xgb = []  # to store logloss of xgboost
ll_lgb = []  # to store logloss of lightgbm
ll_stacking = []  # to store logloss of stacking
ll_voting = [] # to store logloss of voting
ac_rf = []   # to store logloss of random forest
ac_xgb = []  # to store logloss of xgboost
ac_lgb = []  # to store logloss of lightgbm
ac_stacking = []  # to store logloss of stacking
ac_voting = [] # to store logloss of voting


#Same code as above for generating the dataset, applying the 3-layer learning architecture and copmparing with  
#LogisticRegression and GradientBoosting based ensembles. 
#The code is applied to each independent problem/dataset (each dataset with a different number of classes).
for i in classes:
    print('Computing log-loss of the dataset with n_classes: %s, please be patient...' %(i))
    n_classes = i
    
    random_state = 0
    
    # Generating the data
    data, labels = make_classification(n_samples = 2000, n_features = 100, 
                                       n_informative = 50, n_classes = n_classes, random_state = random_state)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = random_state)
    

    # Xgboost
    clf_xgb = XGBClassifier(objective = 'multi:softprob', learning_rate = 0.08, 
                            max_depth = 7, n_estimators = 100,
                            subsample = 0.6, colsample_bytree = 0.8)

    # Random Forest
    n_estimators_rf = 150
    n_jobs_rf = 20
    max_depth_rf = 20
    min_samples_leaf_rf = 30
    min_samples_split_rf = 20
    clf_rf = RandomForestClassifier(n_estimators = n_estimators_rf, n_jobs = n_jobs_rf,
                                  max_depth = max_depth_rf, min_samples_leaf = min_samples_leaf_rf,
                                  min_samples_split = min_samples_split_rf, random_state = 0,
                                  bootstrap = True)
    
    # LightGBM
    clf_lgb = lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.08,
                                max_bin = 255, max_depth = 16, min_child_samples = 10, min_child_weight = 5,
                                min_split_gain = 0.0, n_estimators = 200, objective = 'multiclass',
                                random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42, 
                                silent = True, subsample = 1.0,
                                subsample_for_bin = 50000, subsample_freq = 1)
    
    # Voting classifier
    clf_voting = VotingClassifier(estimators=[('xgb', clf_xgb), ('rf', clf_rf), ('lgb', clf_lgb)], 
                              voting = 'soft', weights = [4, 2, 3])
    
    # meta classifier is decision tree
    max_depth_dt = 10
    min_samples_leaf_dt = 30
    min_samples_split_dt = 6
    meta_clf = CalibratedClassifierCV(DecisionTreeClassifier(criterion = 'gini', max_depth = max_depth_dt, 
                                                             min_samples_leaf = min_samples_leaf_dt,
                                                             min_samples_split = min_samples_split_dt), method = 'isotonic', cv = 5)



    clf_stacking = StackingClassifier(classifiers = [clf1, clf2, clf3], meta_classifier = CalibratedClassifierCV(LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'), method = 'isotonic', cv = 5))
    
    # xgboost logloss score
    clf_xgb.fit(X_train, y_train)
    predict_xgb = clf_xgb.predict_proba(X_test)
    ll_xgb.append(log_loss(y_test, predict_xgb))   # Saving the logloss score
    prediction_xgb = clf_xgb.predict(X_test)
    ac_xgb.append(accuracy_score(y_test, prediction_xgb))

    # random forest logloss score
    clf_rf.fit(X_train, y_train)
    predict_rf = clf_rf.predict_proba(X_test)
    ll_rf.append(log_loss(y_test, predict_rf))   # Saving the logloss score
    prediction_rf = clf_rf.predict(X_test)
    ac_rf.append(accuracy_score(y_test, prediction_rf))
    
    # lightgbm logloss score
    clf_lgb.fit(X_train, y_train)
    predict_lgb = clf_lgb.predict_proba(X_test)
    ll_lgb.append(log_loss(y_test, predict_lgb))   # Saving the logloss score
    prediction_lgb = clf_lgb.predict(X_test)
    ac_lgb.append(accuracy_score(y_test, prediction_lgb))
    
    # voting logloss score
    clf_voting.fit(X_train, y_train)
    predict_voting = clf_voting.predict_proba(X_test)
    ll_voting.append(log_loss(y_test, predict_voting))   # Saving the logloss score
    prediction_voting = clf_voting.predict(X_test)
    ac_voting.append(accuracy_score(y_test, prediction_voting))
    
    # stacking logloss score
    clf_stacking.fit(X_train, y_train)
    predict_stacking = clf_stacking.predict_proba(X_test)
    ll_stacking.append(log_loss(y_test, predict_stacking))   # Saving the logloss score
    prediction_stacking = clf_stacking.predict(X_test)
    ac_stacking.append(accuracy_score(y_test, prediction_stacking))
    
    print('')
    
ac_xgb = np.array(ac_xgb) 
ac_rf = np.array(ac_rf) 
ac_lgb = np.array(ac_lgb)
ac_voting = np.array(ac_voting) 
ac_stacking = np.array(ac_stacking)

# plot the figure
plt.figure(figsize=(10,7))
plt.plot(classes, ac_xgb, 'bo-', label='Xgboost')
plt.plot(classes, ac_rf, 'mo-', label='Random Forest')
plt.plot(classes, ac_lgb, 'yo-', label='LightGBM')
plt.plot(classes, ac_voting, 'go-', label='voting')
plt.plot(classes, ac_stacking, 'ro-', label='Stacking')

plt.title('Log-loss of the different models for different number of classes.')
plt.xlabel('Number of classes')
plt.ylabel('Log-loss')
plt.grid(True)
plt.legend(loc=4)
plt.show()


# In[56]:


# plot the figure
plt.figure(figsize=(10,7))
plt.plot(classes, ll_xgb, 'bo-', label='Xgboost')

plt.plot(classes, ll_lgb, 'yo-', label='LightGBM')

plt.plot(classes, ll_stacking, 'ro-', label='Stacking')

plt.title('Log-loss of the different models for different number of classes.')
plt.xlabel('Number of classes')
plt.ylabel('Log-loss')
plt.grid(True)
plt.legend(loc=4)
plt.show()


# In[54]:


print(ll_xgb)
print(ll_lgb)
print(ll_stacking)


# In[67]:


# plot the histogram of NDCG, accuracy, precision, recall of each classifier

# NDCG
x_model = np.arange(9)
model = ['Log', 'DT', 'NB', 'RF', 'Xgb', 'ETC', 'Ada', 'LightGBM', 'Stacking', 'Voting']
NDCG_model = [NDCG_total_log.mean(), NDCG_total_dt.mean(), NDCG_total_nb.mean(), NDCG_total_rf.mean(), NDCG_total_etc.mean(),
              NDCG_total_xgb.mean(), NDCG_total_ab.mean(), NDCG_total_lgb.mean(), NDCG_total_stacking.mean(), NDCG_total_vote.mean()]
barlist = plt.bar(x_model, NDCG_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of NDCG
NDCG_max = max(NDCG_model)
idx = NDCG_model.index(NDCG_max)
barlist[idx].set_color('g')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('NDCG of each model')
plt.title('Model performance(NDCG)')
plt.show()
print('Model {0} has the highest NDCG score: {1}'.format(model[idx], NDCG_max))


# In[68]:


# plot the histogram of NDCG, accuracy, precision, recall of each classifier

# accuracy
x_model = np.arange(9)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM', 'Stacking', 'Voting']
accuracy_model = [accuracy_log, accuracy_dt, accuracy_nb, accuracy_rf, accuracy_etc, accuracy_xgb, accuracy_ab, accuracy_lgb, accuracy_stacking, accuracy_vote]
barlist = plt.bar(x_model, accuracy_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of accuracy
accuracy_max = max(accuracy_model)
idx = accuracy_model.index(accuracy_max)
barlist[idx].set_color('y')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('Accuracy of each model')
plt.title('Model performance(accuracy)')
plt.show()
print('Model {0} has the highest accuracy score: {1}'.format(model[idx], accuracy_max))


# In[69]:


# plot the histogram of NDCG, precision, precision, recall of each classifier

# precision
x_model = np.arange(9)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM', 'Stacking', 'Voting']
precision_model = [precision_log, precision_dt, precision_nb, precision_rf, precision_etc, precision_xgb, precision_ab, precision_lgb, precision_stacking, precision_vote]
barlist = plt.bar(x_model, precision_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of precision
precision_max = max(precision_model)
idx = precision_model.index(precision_max)
barlist[idx].set_color('indigo')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('Precision of each model')
plt.title('Model performance(precision)')
plt.show()
print('Model {0} has the highest precision score: {1}'.format(model[idx], precision_max))


# In[70]:


# plot the histogram of NDCG, accuracy, precision, recall of each classifier

# recall
x_model = np.arange(9)
model = ['Log', 'DT', 'NB', 'RF', 'ETC', 'Xgb', 'Ada', 'LightGBM', 'Stacking', 'Voting']
recall_model = [recall_log, recall_dt, recall_nb, recall_rf, recall_etc, recall_xgb, recall_ab, recall_lgb, recall_stacking, recall_vote]
barlist = plt.bar(x_model, recall_model, align = 'center', alpha = 0.7)
 
# obtain the highest score of recall
recall_max = max(recall_model)
idx = recall_model.index(recall_max)
barlist[idx].set_color('orange')
plt.xticks(x_model, model)
plt.xlabel('Model')
plt.ylabel('Recall of each model')
plt.title('Model performance(recall)')
plt.show()
print('Model {0} has the highest recall score: {1}'.format(model[idx], recall_max))


# # Step 6: Final Test and Submission

# In[89]:


# define the voting prediction function for final test
# Combine Xgboost, Random Forest, Logistic Regression and LightGBM
# weights are: Xgboost 3, Random Forest 2, Logistic Regression 2, LightGBM 4
# Boom! So exciting!

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def voting_prediction(X_train, y_train, X_test):
    vote_xgb = 5
    vote_rf = 1
    vote_lgb = 2
    prediction = np.zeros((X_test.shape[0], 12))
    
    # Xgboost
    clf1 = XGBClassifier(objective = 'multi:softprob', learning_rate = 0.1, 
                         max_depth = 7, n_estimators = 85, 
                         subsample = 0.6, colsample_bytree = 0.8)

    # Random Forest
    n_estimators_rf = 100
    n_jobs_rf = 20
    max_depth_rf = 20
    min_samples_leaf_rf = 30
    min_samples_split_rf = 20
    clf2 = RandomForestClassifier(n_estimators = n_estimators_rf, n_jobs = n_jobs_rf,
                                  max_depth = max_depth_rf, min_samples_leaf = min_samples_leaf_rf,
                                  min_samples_split = min_samples_split_rf, random_state = 0,
                                  bootstrap = True)

    # LightGBM
    clf3 = lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.08,
                              max_bin = 255, max_depth = 16, min_child_samples = 10, min_child_weight = 5,
                              min_split_gain = 0.0, n_estimators = 100, objective = 'multiclass',
                              random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42, 
                              silent = True, subsample = 1.0,
                              subsample_for_bin = 50000, subsample_freq = 1)
    
    # Xgboost training and testing
    print('Xgboost:')
    for i in range(vote_xgb):
        startTime = time.time()
        print('Iteration:', i + 1)
        clf1.fit(X_train, y_train)
        prediction += clf1.predict_proba(X_test)
        print('Yeah! Xgboost vote', i + 1, 'succeeds! Took %fs!' % (time.time() - startTime))
    print()
    
    # Random Forest training and testing
    print('Random Forest:')
    for i in range(vote_rf):
        startTime = time.time()
        print('Iteration:', i + 1)
        clf2.fit(X_train, y_train)
        prediction += clf2.predict_proba(X_test)
        print('Yeah! Random Forest vote', i + 1, 'succeeds! Took %fs!' % (time.time() - startTime))
    print()
    
    # LightGBM training and testing
    print('LightGBM:')
    for i in range(vote_lgb):
        startTime = time.time()
        print('Iteration:', i + 1)
        clf3.fit(X_train, y_train)
        prediction += clf3.predict_proba(X_test)
        print('Yeah! LightGBM vote', i + 1, 'succeeds! Took %fs!' % (time.time() - startTime))
        
    return prediction


# In[90]:


# define the stacking prediction function for final test
# Combine Xgboost, Random Forest LightGBM
# Meta classifier is the decision tree
# Boom! So exciting!

from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def stacking_prediction(X_train, y_train, X_test):
    print('Stacking training and predicting begins, please wait...')
    startTime = time.time()
    prediction = np.zeros((X_test.shape[0], 12))
    
    # stacking strategy using xgboost, lighgbm and random forest
    # meta classifier is the decision tree
    max_depth_dt = 10
    min_samples_leaf_dt = 30
    min_samples_split_dt = 6
    meta_clf = CalibratedClassifierCV(DecisionTreeClassifier(criterion = 'gini', max_depth = max_depth_dt, 
                                  min_samples_leaf = min_samples_leaf_dt,
                                  min_samples_split = min_samples_split_dt), method = 'isotonic', cv = 5)
    
    # Xgboost
    clf1 = XGBClassifier(objective = 'multi:softprob', learning_rate = 0.1, 
                         max_depth = 7, n_estimators = 85, 
                         subsample = 0.6, colsample_bytree = 0.8)
    
    # LightGBM
    clf2 = lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.08,
                              max_bin = 255, max_depth = 16, min_child_samples = 10, min_child_weight = 5,
                              min_split_gain = 0.0, n_estimators = 100, objective = 'multiclass',
                              random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42, 
                              silent = True, subsample = 1.0,
                              subsample_for_bin = 50000, subsample_freq = 1)
    
    # Random Forest
    n_estimators_rf = 150
    n_jobs_rf = 20
    max_depth_rf = 20
    min_samples_leaf_rf = 30
    min_samples_split_rf = 20
    clf3 = RandomForestClassifier(n_estimators = n_estimators_rf, n_jobs = n_jobs_rf,
                                  max_depth = max_depth_rf, min_samples_leaf = min_samples_leaf_rf,
                                  min_samples_split = min_samples_split_rf, random_state = 0,
                                  bootstrap = True)
    
    stacking_clf = StackingClassifier(classifiers = [clf1, clf2, clf3], meta_classifier = meta_clf)
    
    stacking_clf.fit(X_train, y_train)
    prediction += stacking_clf.predict_proba(X_test)
    print('Yeah! Stacking succeeds! Took %fs!' % (time.time() - startTime))
    print()
    
    return prediction


# In[91]:


from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# prediction begins
train_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/X_extracted.csv', index_col = 0)
test_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_extracted.csv', index_col = 0)
label_encoded = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_encoded.csv', index_col = 0)
label_cols = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/label_cols.csv', index_col = 0)

train_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/train_users_2.csv')
test_users = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_users.csv')

train_extracted.index = train_users['id']
test_extracted.index = test_users['id']

train_test_extracted = pd.concat((train_extracted, test_extracted), axis = 0)   
train_test_scaled = preprocessing.scale(train_test_extracted)     # normalization processing to improve accuracy 
train_test_extracted_new = pd.DataFrame(train_test_scaled, index = train_test_extracted.index)

train_extracted = train_test_extracted_new.ix[train_extracted.index]
test_extracted = train_test_extracted_new.ix[test_extracted.index]

# encode each country
encoder = LabelEncoder()
encoder.fit(label_cols['country_destination'])
prediction = np.zeros((test_extracted.shape[0], 12))

# compute prediction using voting classifier
prediction += voting_prediction(train_extracted, label_encoded, test_extracted)

# In EDA, we could find out that most Airbnb users created account in 2014 and all accounts in test users are created in 2014
# So the weight of 2014 should be larger
# Use only new data for prediction. 
# New data, are considered those that are an year_account_created equal or higher than 2014.
year_new = 2014

train_set = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/X_extracted.csv', index_col = 0)
train_set = train_set[train_set['year_account_created'] >= 2014]
train_set.index = train_users_processed[train_users_processed['year_account_created'] >= 2014].index

test_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_extracted.csv', index_col = 0)
test_extracted.index = test_users['id']

# combine the new train set and the test set together for normalization
train_test_extracted_new = pd.concat((train_set, test_extracted), axis = 0)   
train_test_scaled_new = preprocessing.scale(train_test_extracted_new)     # normalization processing to improve accuracy 
train_test_extracted_new_new = pd.DataFrame(train_test_scaled_new, index = train_test_extracted_new.index)

# obtain the scaled train set and test set
train_set_new = train_test_extracted_new_new.ix[train_set.index]
test_extracted_new = train_test_extracted_new_new.ix[test_extracted.index]
test_extracted_new.columns = train_set_new.columns
label_encoded.index = train_extracted.index
labels_new = label_encoded.ix[train_set.index]['country destination']

# weight of the new training is twice than the original one
print()
for i in range(0, 2):
    print('Prediction for new data: iteration', i + 1)
    prediction += voting_prediction(train_set_new, labels_new, test_extracted_new)
    print()


# In[ ]:


# In EDA, we could find out that most Airbnb users created account in 2014 and all accounts in test users are created in 2014
# So the weight of 2014 should be larger
# Use only new data for prediction. 
# New data, are considered those that are an year_account_created equal or higher than 2014.
year_new = 2014

train_set = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/X_extracted.csv', index_col = 0)
train_set = train_set[train_set['year_account_created'] >= 2014]
train_set.index = train_users_processed[train_users_processed['year_account_created'] >= 2014].index

test_extracted = pd.read_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_extracted.csv', index_col = 0)
test_extracted.index = test_users['id']

# combine the new train set and the test set together for normalization
train_test_extracted_new = pd.concat((train_set, test_extracted), axis = 0)   
train_test_scaled_new = preprocessing.scale(train_test_extracted_new)     # normalization processing to improve accuracy 
train_test_extracted_new_new = pd.DataFrame(train_test_scaled_new, index = train_test_extracted_new.index)

# obtain the scaled train set and test set
train_set_new = train_test_extracted_new_new.ix[train_set.index]
test_extracted_new = train_test_extracted_new_new.ix[test_extracted.index]
test_extracted_new.columns = train_set_new.columns
label_encoded.index = train_extracted.index
labels_new = label_encoded.ix[train_set.index]['country destination']

# weight of the new training is twice than the original one
for i in range(0, 2):
    print('Prediction for new data: iteration', i + 1)
    prediction += stacking_prediction(train_set_new, labels_new, test_extracted_new)
    print()


# In[92]:


# Use the 5 classes with highest scores.
test_extracted.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/test_extracted.csv')
ids, countries = ([], [])
for i in range(test_extracted.shape[0]):
    idx = test_extracted.index[i]
    ids += [idx] * 5
    countries += encoder.inverse_transform(np.argsort(prediction[i])[::-1])[:5].tolist()
submission = pd.DataFrame(np.column_stack((ids, countries)), columns = ['id', 'country'])
submission.to_csv('/Users/mac/Desktop/Python/Python workspace/Datasets/Airbnb New User Bookings/final_submission_voting.csv', index = False)


# In[ ]:


# Support Vector Machine
from sklearn import svm 
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier


predict_total_svm = [0 for k in range(numRows)]
classifier = svm.SVC(kernel = 'linear', random_state = 0)

for iteration in range(0, 10):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test sets
    X_test = X_scaled[subset_index_array[iteration]]
    y_test = y_encoded.loc[subset_index_array[iteration]]
    numRow_range = np.arange(0, numRows, 1)
    train_index = np.delete(numRow_range, subset_index_array[iteration])
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
    
    classifier.fit(X_train, y_train)  
    
    prediction = classifier.predict(X_test)  
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    
    for j in range(len(subset_index_array[iteration])):
        predict_total_svm[subset_index_array[iteration][j]] = prediction[j]    
    
Confusion_matrix_svm = confusion_matrix(y_encoded, predict_total_svm)
Confusion_matrix_svm_dataframe = pd.DataFrame(data = Confusion_matrix_svm, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_svm_dataframe)
print()

print('Accuracy score:')  
print(accuracy_score(y_encoded, predict_total_svm))  
print()

print('The report of classification using support vector machine: ')
print(classification_report(y_encoded, predict_total_svm))  


# In[ ]:


# Gradient Tree Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)

# Predictions using gradient boosting
max_depth_gb = 10
min_samples_leaf_gb = 30
min_samples_split_gb = 6
n_estimators_gb = 150
learning_rate_gb = 0.2
classifier = GradientBoostingClassifier(learning_rate = learning_rate_gb, n_estimators = n_estimators_gb, 
                                        min_samples_split = min_samples_split_gb, min_samples_leaf = min_samples_leaf_gb, 
                                        max_depth = max_depth_gb, random_state = 0)

predict_total_gb = []
iteration = 0
for train_index, test_index in kf.split(X_extracted, y_encoded):
    startTime = time.time()
    print('Iteration:', iteration + 1)
    
    # determine training sets and the test set
    X_test = X_scaled[test_index]
    y_test = y_encoded.loc[test_index]
    X_train = X_scaled[train_index]
    y_train = y_encoded.loc[train_index]
     
    classifier.fit(X_train, y_train)  
    
    prediction = classifier.predict(X_test)  
    
    print('Accuracy:', accuracy_score(y_test, prediction))    
    print('Iteration', iteration + 1, 'takes %fs!' % (time.time() - startTime)) 
    print()
    for j in range(len(prediction)):
        predict_total_gb.append(prediction[j])
    iteration += 1
    
Confusion_matrix_gb = confusion_matrix(y_encoded, predict_total_gb)
Confusion_matrix_gb_dataframe = pd.DataFrame(data = Confusion_matrix_gb, 
                                              index = ['AU(true)', 'CA(true)', 'DE(true)', 'ES(true)', 'FR(true)', 'GB(true)',
                                                       'IT(true)', 'NDF(true)', 'NL(true)', 'PT(true)', 'US(true)', 'other(true)'],
                                              columns = ['AU(prediction)', 'CA(prediction)', 'DE(prediction)', 'ES(prediction)', 'FR(prediction)', 'GB(prediction)',
                                                         'IT(prediction)', 'NDF(prediction)', 'NL(prediction)', 'PT(prediction)', 'US(prediction)', 'other(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_gb_dataframe)
print()

print('Accuracy score:')  
print(accuracy_score(y_encoded, predict_total_gb))  
print()

print('The report of classification using gradient boosting: ')
print(classification_report(y_encoded, predict_total_gb))  

