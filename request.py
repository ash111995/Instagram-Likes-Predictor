
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'numberPosts':61, 'numberFollowing':522, 'numberFollowers':78689, 'number_tags':3})

print(r.json())