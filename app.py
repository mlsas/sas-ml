#from flask_wtf import FlaskForm
#from flask import Flask, render_template
#from wtforms.validators import DataRequired
#from wtforms import IntegerField, BooleanField, FloatField, SubmitField, RadioField
#import os
#import numpy as np
#from flask import Flask, request, jsonify, render_template


#create an instance of Flask

#import Flask
import pickle
from flask import Flask, render_template, request
import numpy as np
import joblib

'''
import warnings

with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      estimator = joblib.load('model.pkl')
'''
#from flask_bootstrap import Bootstrap

app = Flask(__name__)

#app.config['SQLALCHEMY_DATABASE_URL']='postgres://fmoefpprrbynnu:578bb8c80fb4d480e21cf89cd36b1084f1faa06f8ade7b2c85d4744a192b3853@ec2-34-206-8-52.compute-1.amazonaws.com:5432/df07i66fmaii4i?sslmode=require'

#Bootstrap(app)


@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #get data
        trip = request.form.get('trip')
        load = request.form.get('load')
        frequency = request.form.get('frequency')
        print(trip, load, frequency)

        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(trip, load, frequency)
            #prediction = preprocessDataAndPredict([str(trip), float(load), float(frequency)])
            print(type(trip), type(load), type(frequency))
            print(prediction)
            res = prediction

            def prediction_str():
                if res <= -1:
                    print("Low Fault")
                    return 'Low Fault'
                elif res == 0:
                    print("Medium Fault")
                    return 'Medium Fault'
                else:
                    print("High Fault")
                    return 'High Fault'

            print(prediction_str())

            #pass prediction to template
            return render_template('predict.html', prediction = prediction_str())

        except ValueError:
            return "Please Enter valid values"
        pass
    pass


def preprocessDataAndPredict(trip, load, frequency):
    
    #keep all inputs in array
    test_data = [trip, load, frequency]
    #print(type(trip), type(load), type(frequency))
    print(test_data)
    
    #convert value data into numpy array
    #test_data = np.array(test_data)
    
    #test_data = np.array([trip, load, frequency])
    test_data = np.array([str(trip), float(load), float(frequency)])
    #test_data = np.array([[int(t), float(l), float(f)]])
    
    #reshape array
    test_data = test_data.reshape(1,-1)
    print(test_data)
    
    #declare path where you saved your model
    outFileFolder = '/Users/saseng/AML/My-Projects/Electric Fault/eFault/'
    filePath = outFileFolder + 'E_Fault_model.pkl'
    #open file
    file = open(filePath, "rb")
    #load the trained model
    trained_model = joblib.load(file)
    
    filePath = outFileFolder + 'E_Fault_sc.pkl'
    #open file
    file = open(filePath, "rb")
    #load the trained model
    trained_sc = joblib.load(file)
    
    test_data = trained_sc.transform(test_data)
    print(test_data)
    
    prediction = trained_model.predict(test_data)
    print(prediction)
      
    return prediction
    
    pass  
            
    #pass prediction to template

if __name__ == '__main__':
    app.run(debug=True)


 