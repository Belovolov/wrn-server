##############################
### ML IMAGE DETECTION SERVER
##############################

#############
#### IMPORTS
#############

# Flask
from flask import Flask, render_template, make_response, request, Response, redirect, send_from_directory, jsonify

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
from wr_nets import *
wrn = Wrn168c10("weights/WRN-16-8-Weights.h5")

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

##################
#### FLASK SETUP
##################
app = Flask(__name__)

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
        if submitted_file:
            ### do prediction
            preds = wrn.get_prediction(submitted_file)
            return jsonify(preds)
    else:       
        return render_template('error.html')

#######
### RUN
#######
if __name__ == "__main__":
    # run app
    app.run(host = "0.0.0.0", port = int("8082"))