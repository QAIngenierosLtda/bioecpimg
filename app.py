"""
API para verificacion automatica de imagenes para deteccion Biometrica 
Usage:
    API endpoint Tipo POST
    http://bio01.qaingenieros.com/api/img
    Envia MultiType Form
    file: [image file]
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
import json
import datetime
import os
from os import path
from os.path import dirname
# import glob
import numpy as np
import cv2
import logging

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask,  request, make_response, render_template, send_from_directory
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS



# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash
# import uuid
# import jwt



# from PIL import Image


app = Flask(__name__)
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
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

result = {}
motivos = []

# Load Neuronal Net
os.path.join(app.config['UPLOAD_FOLDER'], 'imagen.jpg')

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.join(dirname(__file__), "deploy.prototxt")
weightsPath = os.path.join(dirname(__file__),
	"res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model('mask_detector.model')


@app.route('/uploads/<path:filename>')
def upload_file(filename):
    
    
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True, cache_timeout=0)

@app.route('/crops/<path:filename>')
def result_file(filename):
    return send_from_directory(app.config['PROCESS_FOLDER'], filename, as_attachment=True, cache_timeout=0)

# Detecta mascara
def detect_mask(image):
    # Detecta el tamaño de la imagen
    (h, w) = image.shape[:2]

    #construye un Blob de la imagen
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

    # envia el blob a traves de la red neuronal y detecta la mascara
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    maskStat = "No Mask"

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            print(mask, withoutMask)
            # determine the class label and color we'll use to draw
            # the bounding box and text
            maskStat = "Mask" if mask > withoutMask else "No Mask"
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            cv2.imwrite('crops/' + 'mask.jpg', image)

    return maskStat


# Escala la imagen si se requiere
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# Recorta la imagen alrededor de la cara detectada
def crop_image(filename):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    # Read the input image
    img = cv2.imread(filename)
    height, width, channels = img.shape

    print("Image for crop charateristics: W:" , width , " H:" , height)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor = 1.2, 
        minNeighbors = 5,
        minSize = (200,200)
    )

    # print(filename)
    # print(faces)
    print("Face coordinates: " , faces)
    # Draw rectangle around the faces
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Graba la imagen con el rectangulo
        cv2.imwrite('crops/' + 'salida.jpg', img)

        rw = w/width
        rh = h/height

        of_x = (int)((w*rw/0.5)/4)
        of_y = (int)((h*rh/0.2)/4)


        print("Relaciones de ancho y alto con imagen Rw, Rh", rw, rh )
        print("Ofset de ancho y alto con imagen of_x, of_y", of_x, of_y )

        x1 = x - of_x
        if (x1 < 0):
            x1 = 0
        y1 = y - of_y
        if (y1 < 0):
            y1 = 0
        x2 = x+w+of_x
        if (x2 > width):
            x2 = width
        y2 = y+h+of_y
        if (y2 > height):
            y2 = height
    # for (x, y, w, h) in faces:
    #     x1= x-int(w*0.4)
    #     if x1 < 0:
    #         x1 = 0
    #     y1 = y-int(h*1.3)
    #     if y1 < 0:
    #         y1 = 0
    #     x2 = x+w+int(y*0.4)
    #     if x2 > width:
    #         x2 = width
    #     y2 = y+h+int(h*1.3)   
    #     if y2 > height:
    #         y2 = height    
        
        print("Coordenadas recorte" , y1,y2,x1,x2)

        faceimg = img[y:y+h,x:x+w]
        cv2.imwrite('crops/face.jpg', faceimg)

        img2 = img[y1:y2,x1:x2]
        cv2.imwrite('crops/frame.jpg', img2)

        img2 = image_resize(img2, width=640)
        # write the output
        print("Escribiendo imagen recortada")
        cv2.imwrite('crops/crop.jpg', img2)

# Detecta las caras que esten en la imagen        
def detect_faces(img):
    # Get image sizes
    # height, width, channels = img.shape
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('crops/' + 'gray.jpg', gray)
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor = 1.2, 
        minNeighbors = 5,
        minSize = (200,200)
    )
    print ("Found {0} faces!".format(len(faces)))

    faceImg = img
    # # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(faceImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imwrite('crops/' + 'faces.jpg', faceImg)

    return faces

# Determinar las caracteristicas de W,H y canales
def test_size(img):
    height, width, channels = img.shape
    print("Image charateristics: W:" , width , " H:" , height)
    return { "height": height, "width": width, "channels": channels }

# Verifica la varianza (componentes de baja frecuencia) para analizar el enfoque
def test_blur(img):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(img, cv2.CV_64F).var()

# calculate Blurriness (Desenfoque)
def detect_blur_fft(image, size=60, thresh=10):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply 
    # the inverse FFT

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)

# Funcion que ejecuta la verificacion de imagenes
def test_image(filename):
    img = cv2.imread(filename)
    # height, width, channels = img.shape

    # Definicion de los valores iniciales
    result["status"] = "aprobado"
    result["motivos"] = []
    result["original_size"] = test_size(img)
    result["final_size"] = {}
    result["blur"] = test_blur(img)

    #Convierte la imagen a tonos de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, size=60,thresh = 10)
    result["mean_fft"] = mean
    print(blurry)
    result["blurry"] = 0
    if (blurry):
        result["blurry"] = 1
        
    # result["blurry_fft"] = blurry

    result["faces"] = 0
    result["image"] = ""
    motivos = []
    
    # detecta numero de caras en la imagen
    faces = detect_faces(img)
    result["faces"] = len(faces)

    result["mask"] = detect_mask(img)

    print("Mascara : " , result["mask"])
    # Si se detecta una mascara se rechaza la imagen
    if (result["mask"] == "Mask"):
        motivos.append("No se permite usar mascara para la foto")
        result["status"] = "rechazado"

    # Si se detecta mas de una cara se rechaza
    if len(faces) > 1:
        motivos.append("Mas de una cara")
        result["status"] = "rechazado"

    # Si NO se detecta cara valida se rechaza
    if len(faces) == 0:
        motivos.append("No se detecto una cara valida")
        result["status"] = "rechazado"

    # Si la varianza es menor a 100 define imagen borrosa -> rechaza
    if result["blurry"] == 1:
        motivos.append("Imagen desenfocada")
        result["status"] = "rechazado"
     
    # Si la imagen es muy grande la reduce a un ancho de 2048 (para poder procesarla)
    if (result["original_size"]["width"] > 2048):
        img = image_resize(img, width=2048)
        cv2.imwrite(filename, img)

    # Si la imagen tiene menos de 640x480 (o uno de los dos) se rechaza por resolucion
    elif ((result["original_size"]["width"] < 640) or (result["original_size"]["height"] < 800)):
        motivos.append("Imagen pequena")
        result["status"] = "rechazado"
        
    # Si cumple las condiciones recorta la imagen alrededor de la cara
    if img is not None and result["status"] == "aprobado":        
        crop_image(filename)
        import base64
        if (path.exists("crops/face.jpg")):
            imgOut = cv2.imread("crops/crop.jpg")
            result["final_size"] = test_size(imgOut)
            string = base64.b64encode(cv2.imencode('.jpg', imgOut)[1]).decode()
            result["image"] = string
        else:
            result["status"] = "rechazado"
            motivos.append("No fue posible crear recorte de imagen")
            result["image"] = ""

    # Si la imagen se ha rechazado , se actualiza los motivos
    if result["status"] != "aprobado":
        result["motivos"]= motivos

    with open('crops/response.json', 'w') as json_file:
        json.dump(result, json_file)

    # Se retorna el JSON del resultado
    return result


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class data(Resource):
    def get(self):
        #return "Welcome!"
        # return render_template('index.html')
        response = make_response(render_template('index.html'))
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

        # Se realiza un POST al endpoint
        file = request.files['file']
        print(file)
        res = "No image sent :("
        # print(file)
        if file:
            # Existe un archivo en el POST
            if file and allowed_file(file.filename):
                filename = file.filename
                print(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'imagen.jpg'))
                if (path.exists("crops/face.jpg")):
                    os.remove('crops/face.jpg')
                if (path.exists("crops/faces.jpg")):
                    os.remove('crops/faces.jpg')
                if (path.exists("crops/crop.jpg")):
                    os.remove('crops/crop.jpg')
                res = test_image(os.path.join(app.config['UPLOAD_FOLDER'], 'imagen.jpg'))


        return res

# Definicion del EndPoint
api.add_resource(ProcessImageEndpoint, '/')
api.add_resource(data,'/images')

if __name__ == '__main__':
  app.run(debug=True)
