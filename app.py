from flask import Flask,  request, jsonify, make_response
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash
# import uuid
# import jwt
import datetime
# from functools import wraps
import os
import glob
import cv2


# from PIL import Image


app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['SECRET_KEY']='S3cr3t'
app.config['SQLALCHEMY_DATABASE_URI']='sqlite://///Users/amejia/Proyectos/QA/bioecpimg/library.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# db = SQLAlchemy(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

result = {}
motivos = []

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

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor = 1.2, 
        minNeighbors = 5,
        minSize = (100,100)
    )

    # print(filename)
    # print(faces)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        x1= x-int(w*0.4)
        if x1 < 0:
            x1 = 0
        y1 = y-int(h*0.5)
        if y1 < 0:
            y1 = 0
        x2 = x+w+int(y*0.4)
        if x2 > width:
            x2 = width
        y2 = y+h+int(h*0.5)   
        if y2 > height:
            y2 = height    
        img2 = img[y1:y2,x1:x2]
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
        minSize = (100,100)
    )
    print ("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imwrite('crops/' + 'salida.jpg', img)

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
    result["faces"] = 0
    result["image"] = ""
    motivos = []
    
    # detecta numero de caras en la imagen
    faces = detect_faces(img)
    result["faces"] = len(faces)

    # Si se detecta mas de una cara se rechaza
    if len(faces) > 1:
        motivos.append("Mas de una cara")
        result["status"] = "rechazado"

    # Si NO se detecta cara valida se rechaza
    if len(faces) == 0:
        motivos.append("No se detecto una cara valida")
        result["status"] = "rechazado"

    # Si la varianza es menor a 100 define imagen borrosa -> rechaza
    if result["blur"] < 100:
        motivos.append("Imagen desenfocada")
        result["status"] = "rechazado"
     
    # Si la imagen es muy grande la reduce a un ancho de 2048 (para poder procesarla)
    if (result["original_size"]["width"] > 2048):
        img = image_resize(img, width=2048)
        cv2.imwrite(filename, img)

    # Si la imagen tiene menos de 640x480 (o uno de los dos) se rechaza por resolucion
    elif ((result["original_size"]["width"] < 480) or (result["original_size"]["height"] < 640)):
        motivos.append("Imagen pequena")
        result["status"] = "rechazado"
        
    # Si cumple las condiciones recorta la imagen alrededor de la cara
    if img is not None and result["status"] == "aprobado":        
        crop_image(filename)
        import base64
        imgOut = cv2.imread("crops/crop.jpg")
        result["final_size"] = test_size(imgOut)
        string = base64.b64encode(cv2.imencode('.jpg', imgOut)[1]).decode()
        result["image"] = string

    # Si la imagen se ha rechazado , se actualiza los motivos
    if result["status"] != "aprobado":
        result["motivos"]= motivos

    # Se retorna el JSON del resultado
    return result


def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
          res = test_image(os.path.join(app.config['UPLOAD_FOLDER'], 'imagen.jpg'))

      return res

# Definicion del EndPoint
api.add_resource(ProcessImageEndpoint, '/')

if __name__ == '__main__':
  app.run(debug=True)