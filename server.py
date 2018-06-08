##############################
### ML IMAGE DETECTION SERVER
##############################

#############
#### IMPORTS
#############

# Flask
from flask import Flask, render_template, make_response, request, Response, redirect, send_from_directory, jsonify
from flask_cors import CORS

# Flask Misc
from flask import request
import json

# Misc
import datetime, time
import os, sys
import shutil

print('### start server ' + str(datetime.datetime.now()))

# Allow relative imports for keras wrapper modules in /utils/
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
                
#import the net
from wrn168c10 import *
predictor = get_wrn168c10predictor("weights/WRN-16-8-Weights.h5")

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

##################
#### FLASK SETUP
##################
app = Flask(__name__)
CORS(app)

# app config
app.config['DEBUG'] = True

app.static_folder = 'static/'
app.static_url_path = 'static/'

###################
### FLASK - ROUTES
###################

# Return dash page
@app.route('/')
def index():
    return jsonify({'status':'OK'})

# POST IMAGE 
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method =='POST':
        submitted_file = request.files['sampleImage'].read()
        print(request.form)
        #print(list(request.form.keys())[0])
        #print(request.form.get('sampleImage'))
        #print(request.data)
        #print(request.files)
        print(request.files['sampleImage'])
        #print(submitted_file)
        print(submitted_file)
        if submitted_file:
            ### do prediction
            preds = predictor(submitted_file)

            #response.headers["Access-Control-Allow-Origin"] = "*"
            print(preds)
            return jsonify(preds)
    else:       
        return render_template('error.html')

#######
### RUN
#######
if __name__ == "__main__":
    # run app
    app.run(host = "0.0.0.0", port = int("8082"))