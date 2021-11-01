#!/usr/bin/env python
# coding: utf-8

# In[13]:


import requests


# In[14]:


url = 'http://172.24.76.92:9697/predict'


# In[15]:


customer_id = 11428
customer_data = {
    "age" : 59,
    "job" : "technician",
    "marital" : "married",
    "education" : "secondary",
    "balance" : 2528,
    "housing": 0,
    "loan": 0,
    "contact": "unknown",
    "day" : 19,
    "month" : "jun",
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}


# In[16]:


print("Requesting service.................")
print("Sending Customer Data for Customer Id : ",customer_id,"  to Term Deposit Subscription Service.......")


# In[17]:


response = requests.post(url, json=customer_data).json()
print(response)


# In[25]:


print("Getting response from the service.................")
print("The probablity of subscription for customer with id : ", customer_id," is", round(response['subscription-probablity'],3))


# In[26]:


print("Suggested input for Marketing Team:")
if(response['subscribe'] == True):
    print("Send marketing email to the potential customer with id", customer_id, " for the subscription of Term Deposit")
else:
    print("Do not Send marketing email to the potential customer with id", customer_id, " for the subscription of Term Deposit")


# In[ ]:




