
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import instagram_explore as insta


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index0.html')


@app.route('/predict',methods=['POST'])
def predict():

    '''
    For rendering results on HTML GUI
    '''

    def inst(uname):
        res = insta.user(uname)
        followers = res.data.get("edge_followed_by").get("count")
        following = res.data.get("edge_follow").get("count")
        Post =  res.data.get("edge_owner_to_timeline_media").get("count")
        l = list([Post, following, followers])
        return l

    username = request.form['username']
    data = inst(username)
    int_features = [np.log(int(x)) for x in data]
    #int_features = [np.log(int(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = np.exp(model.predict(final_features))

    output = round(prediction[0])
    return render_template('index0.html', prediction_text='Likes in future post might be {}'.format(output))




"""
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
"""
if __name__ == "__main__":
    app.run(debug=True)
