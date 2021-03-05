
import json
import datetime
import os
from os import path
from os.path import dirname
from datetime import datetime
import time
# import glob
# import numpy as np
import cv2
import logging
import base64
from PIL import Image
from io import BytesIO

from impr import Imp

# from multiprocessing import Value

# from waitress import serve
# import pdb; pdb.set_trace()

# # si se usa deteccin de mascara descomentar las siguiente 3 linieas
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model


from flask import Flask,  request, make_response, render_template, send_from_directory
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS



# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash
# import uuid
# import jwt



# from PIL import Image

app = Flask(__name__)

a = []
help_message = """
API para verificacion automatica de imagenes para deteccion Biometrica 
Uso:
    API endpoint Tipo POST
        base_url/api/img
        Envia MultiType Form
        file: [image file]
        doc: Numero documento identificacion
    API endpoint Tipo POST
        base_url/api/img/base64
        Envia Application Json {"img":"imagen:base64", "doc":"Numero documento identificacion"}
Returns:
    [json]: 
    {
        "status": "aprobado",
        "motivos": [],
        "original_size": {
            "height": 2012,
            "width": 1470,
            "channels": 3
        },
        "final_size": {
            "height": 875,
            "width": 640,
            "channels": 3
        },
        "blur": 16.006692414047752,
        "mean_fft": 13.887649674324319,
        "blurry": 0,
        "faces": 1,
        "image": "/9j"
    }
"""


api = Api(app)
CORS(app)

app.config['SECRET_KEY']='S3cr3t'
app.config['SQLALCHEMY_DATABASE_URI']='sqlite://///Users/amejia/Proyectos/QA/bioecpimg/library.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

# db = SQLAlchemy(app)

app.config['DETECTOR_FOLDER'] = 'face_detector/'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESS_FOLDER'] = 'crops/'
app.config['ALLOWED_EXTENSIONS'] = { 'png', 'jpg', 'jpeg', 'gif'}

logging.getLogger('flask_cors').level = logging.DEBUG


result = {}
motivos = []

# os.path.join(app.config['UPLOAD_FOLDER'], 'imagen.jpg')

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.join(dirname(__file__), "deploy.prototxt")
weightsPath = os.path.join(dirname(__file__),
	"res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
# print("[INFO] loading face mask detector model...")
# modelPath = os.path.join(dirname(__file__), 'mask_detector.model')
# model = load_model(modelPath)

@app.route('/ayuda', methods=['GET'])
def help():
    return help_message

@app.route('/uploads/<path:filename>')
def upload_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True, cache_timeout=0)

@app.route('/crops/<path:filename>')
def result_file(filename):
    return send_from_directory(app.config['PROCESS_FOLDER'], filename, as_attachment=True, cache_timeout=0)


@app.route('/base64', methods=['POST']) 
def upload_base64_file(): 
    """ 
        Upload image with base64 format and get car make model and year 
        response 
    """

    data = request.get_json()
    # print(data)

    if data is None:
        print("No valid request body, json missing!")
        res = {"success": False, "msg": "No image sent :("}
        return res
    else:
        img_data = data['img']
        gfh = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        doc = data["doc"] + '_' + gfh

        
        img_data = img_data[img_data.find(",")+1:]
        im = Image.open(BytesIO(base64.b64decode(img_data)))
        im.save(os.path.join(app.config['UPLOAD_FOLDER'], doc + '.jpg'));
        filename = doc + '.jpg'

        ip = Imp(filename, doc)
        res = ip.test_image()

        # res = imageprocessing.test_image(os.path.join(app.config['UPLOAD_FOLDER'], filename), doc)

        return res




def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class data(Resource):
    def get(self):
        #return "Welcome!"
        # return render_template('index.html')
        document = request.values["doc"]
        image_name = document + '.jpg'
        faces_name = 'faces_' + document + '.jpg'
        crops_name = 'crop_' + document + '.jpg'
        response = make_response(render_template('index.html', document=document, image_name=image_name, faces_name=faces_name, crops_name=crops_name))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Cache-Control'] = 'public, max-age=0'
        return response

        # headers = {'Content-Type': 'text/html'}
        # upload_image = os.path.join('uploads', 'image.jpg')
        # response = make_response(render_template('index.html'),200,headers)
        # response.cache_control.no_cache = True
        # # response["Cache-Control"] = "no-cache, no-store, must-revalidate" # HTTP 1.1.
        # # response["Pragma"] = "no-cache" # HTTP 1.0.
        # # response["Expires"] = "0" # Proxies.
        # return response

        # 
        # return Response(render_template('index.html'),mimetype='text/html', upload_image=full_filename)


class ProcessImageEndpoint(Resource):
  
    def __init__(self):
        # Create a request parser
        parser = reqparse.RequestParser()
        parser.add_argument("image", type=str,
                            help="Base64 encoded image string", required=True, location='json')
        self.req_parser = parser


    # This method is called when we send a POST request to this endpoint
    def post(self):

        gfh = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        # Se realiza un POST al endpoint
        file = request.files['file']
        # doc = request['doc']
        doc = request.values["doc"] + '_' + gfh
        # print (request)
        print(file)
        # print(doc)

        res = {"success": False, "msg": "No image sent :("}
        # print(file)
        if file:
            # Existe un archivo en el POST
            if file and allowed_file(file.filename):
                #filename = file.filename
                filename = doc + '.jpg'
                print(filename)
                # print(doc)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                filename = filename

                ip = Imp(filename, doc)
                res = ip.test_image()

        return res

# Definicion del EndPoint
api.add_resource(ProcessImageEndpoint, '/')
api.add_resource(data,'/images')

if __name__ == '__main__':
    # serve(app, host="0.0.0.0", port=5000)
    app.run(debug=True)
