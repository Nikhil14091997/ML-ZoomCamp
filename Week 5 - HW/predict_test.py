import requests
url = 'http://localhost:9696/predict_flask'
customer_id = 'xyz-123'
customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}
response = requests.post(url, json=customer).json()
print(response)
if (response['Churn'] == True):
    print(f'sending promotional email to {customer_id}')
else:
    print(f'not sending promotional email to {customer_id}')