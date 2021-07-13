#importing libraries
import numpy as np
from flask import Flask, render_template,request
import pickle

from numpy.core.fromnumeric import round_ 
app = Flask(__name__) #Initialize the flask App
app.static_folder='static'
Nmodel = pickle.load(open('./Trained_Models/N.pkl', 'rb'))
#Pmodel = pickle.load(open('./Trained_Models/P.pkl', 'rb'))
#Kmodel = pickle.load(open('./Trained_Models/K.pkl', 'rb'))
#cropmodel = pickle.load(open('./Trained_Models/crop.pkl', 'rb'))
#yieldmodel = pickle.load(open('./Trained_Models/yield.pkl', 'rb'))
#costmodel = pickle.load(open('./Trained_Models/cost.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [x for x in request.form.values()]
    for i in range(4):
        int_features[i] = float(int_features[i])
    #int_features = [90,42,43] + int_features
    final_features = np.array([int_features[:4]])
    prediction = Nmodel.predict(final_features)[0]
    output = round(prediction) 
    return render_template('index.html', prediction_text='Nitrogen content :{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)