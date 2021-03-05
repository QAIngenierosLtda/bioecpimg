import numpy as np
import cv2
import logging
import base64
from PIL import Image
from io import BytesIO
import time
import os
from os import path
from os.path import dirname
import json

DEBUG_IMG = os.environ.get("DEBUG_IMG")
DEBUG_IMGS = os.environ.get("DEBUG_IMGS")
DEBUG_JSON = os.environ.get("DEBUG_JSON")
DEBUG_IMPR = os.environ.get("DEBUG_IMPR")
class Imp:
  
    def __init__(self, filename, doc, image_name):
        self.filename = 'uploads/' + filename
        self.doc = doc
        self.document = doc
        self.original_name = image_name
        # self.result = []

    def convert_and_save(self, b64_string):
        b64_string += '=' * (-len(b64_string) % 4)  # restore stripped '='s
        string = b'{b64_string}'

    # Detecta mascara
    def detect_mask(self, image, doc):
        # Detecta el tamaño de la imagen
        (h, w) = image.shape[:2]

        #construye un Blob de la imagen
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

        # envia el blob a traves de la red neuronal y detecta la mascara
        if (DEBUG_IMPR):
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

                if (DEBUG_IMPR):
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
                # cv2.putText(image, label, (startX, startY - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                # cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

                cv2.imwrite('crops/' + 'mask_' + doc + '.jpg', image)

        return maskStat


    # Escala la imagen si se requiere
    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
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
    def crop_image(self, filename, doc):
        
        # Load the cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

        # Read the input image
        img = cv2.imread(self.filename)
        height, width, channels = img.shape

        if (DEBUG_IMPR):
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

        eyes = eye_cascade.detectMultiScale(gray)
        print(eyes)

        # print(filename)
        # print(faces)
        if (DEBUG_IMPR):
            print("Face coordinates: " , faces)

        if (DEBUG_IMGS): 
            # Draw rectangle around the faces
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Graba la imagen con el rectangulo
            cv2.imwrite('crops/' + 'salida_' + doc + '.jpg', img)

        rw = w/width
        rh = h/height

        of_x = (int)((w*rw/0.5)/4)
        of_y = (int)((h*rh/0.2)/4)


        if (DEBUG_IMPR):
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
        
        if (DEBUG_IMPR):
            print("Coordenadas recorte" , y1,y2,x1,x2)

        if (DEBUG_IMGS):
            faceimg = img[y:y+h,x:x+w]
            cv2.imwrite('crops/face_' + doc + '.jpg', faceimg)

            img2 = img[y1:y2,x1:x2]
            cv2.imwrite('crops/frame_' + doc + '.jpg', img2)

        img2 = self.image_resize(img2, width=640)

        # write the output
        if (DEBUG_IMPR):
            print("Escribiendo imagen recortada")
        cv2.imwrite('crops/crop_' + doc + '.jpg', img2)

    # Detecta las caras que esten en la imagen        
    def detect_faces(self, img, doc):
        # Get image sizes
        # height, width, channels = img.shape
        # Load the cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (DEBUG_IMG):
            cv2.imwrite('crops/' + 'gray_' + doc + '.jpg', gray)
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor = 1.2, 
            minNeighbors = 5,
            minSize = (200,200)
        )
        if (DEBUG_IMPR):
            print ("Found {0} faces!".format(len(faces)))

        if (DEBUG_IMG):
            faceImg = img
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(faceImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            cv2.imwrite('crops/' + 'faces_' + doc + '.jpg', faceImg)

        return faces

    # Detecta los ojos que esten en la imagen        
    def detect_eyes(self, img, doc):
        # Get image sizes
        # height, width, channels = img.shape
        # Load the cascade
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (DEBUG_IMG):
            cv2.imwrite('crops/' + 'gray_' + doc + '.jpg', gray)
        # Detect faces
        eyes = eye_cascade.detectMultiScale(gray,1.3,11)
        if (DEBUG_IMPR):
            print ("Found {0} eyes!".format(len(eyes)))

        if (DEBUG_IMG):
            faceImg = img
            # Draw a rectangle around the faces
            for (x, y, w, h) in eyes:
                cv2.rectangle(faceImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            cv2.imwrite('crops/' + 'eyes_' + doc + '.jpg', faceImg)

        return eyes

    # Detecta los ojos que esten en la imagen        
    def detect_mouth(self, img, doc):
        # Get image sizes
        # height, width, channels = img.shape
        # Load the cascade
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        test = mouth_cascade.load('haarcascade_mcs_mouth.xml')
        print('Load:' , test)

        # mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'Mouth.xml')
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        # test = face_cascade.load('haarcascade_mcs_mouth.xml')
        # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
        # Convert into grayscale
        # print(mouth_cascade)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (DEBUG_IMG):
            cv2.imwrite('crops/' + 'gray_' + doc + '.jpg', gray)
        # Detect faces
        mouth = mouth_cascade.detectMultiScale(gray,1.7,11)
        
        if (DEBUG_IMPR):
            print ("Found {0} eyes!".format(len(mouth)))

        if (DEBUG_IMPR):
            faceImg = img
            # Draw a rectangle around the faces
            for (x,y,w,h) in mouth:
                y = int(y - 0.15*h)
                cv2.rectangle(faceImg, (x,y), (x+w,y+h), (0,255,0), 3)
                break   
            
            cv2.imwrite('crops/' + 'mouth_' + doc + '.jpg', faceImg)

        return mouth

    # Determinar las caracteristicas de W,H y canales
    def test_size(self, img):
        height, width, channels = img.shape
        if (DEBUG_IMPR):
            print("Image charateristics: W:" , width , " H:" , height)
        return { "height": height, "width": width, "channels": channels }

    # Verifica la varianza (componentes de baja frecuencia) para analizar el enfoque
    def test_blur(self, img):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(img, cv2.CV_64F).var()

    # calculate Blurriness (Desenfoque)
    def detect_blur_fft(self, image, size=60, thresh=10):
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
    # , filename, doc
    def test_image(self):
        img = cv2.imread ( self.filename)
        if (DEBUG_IMPR):
            print(self.filename)
            print(self.doc)
        # height, width, channels = img.shape
        result = {}
        # Definicion de los valores iniciales
        result["infile"] = self.filename
        result["outfile"] = ""
        result["original_name"] = self.original_name
        result["status"] = "aprobado"
        result["motivos"] = []
        result["original_size"] = self.test_size(img)
        result["final_size"] = {}
        result["blur"] = self.test_blur(img)
        result["process_time"] = 0
        result["faces"] = 0
        result["eyes"] = 0
        result["mouth"] = 0
        # result["start_time"] = datetime.now()
        
        start = time.time()

        #Convierte la imagen a tonos de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (mean, blurry) = self.detect_blur_fft(gray, size=60,thresh = 6)
        result["mean_fft"] = mean
        if (DEBUG_IMPR):
            print(blurry)
        result["blurry"] = 0
        if (blurry):
            result["blurry"] = 1
            
        # result["blurry_fft"] = blurry

        result["image"] = ""
        motivos = []
        
        # detecta numero de caras en la imagen
        faces = self.detect_faces(img, self.doc)
        result["faces"] = len(faces)
        
        # detecta numero de ojos en la imagen
        eyes = self.detect_eyes(img, self.doc)
        result["eyes"] = len(eyes)

        # detecta la boca en la imagen
        mouth = self.detect_mouth(img, self.doc)
        result["mouth"] = len(mouth)

        # Si se detecta mas de una cara se rechaza
        if len(eyes) != 2:
            motivos.append("No se detectaron los dos ojos")
            result["status"] = "rechazado"
        # result["mask"] = detect_mask(img, doc)

        # print("Mascara : " , result["mask"])
        # # Si se detecta una mascara se rechaza la imagen
        # if (result["mask"] == "Mask"):
        #     motivos.append("No se permite usar mascara para la foto")
        #     result["status"] = "rechazado"

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
            motivos.append("Imagen desenfocada o bajo contraste")
            result["status"] = "rechazado"
        
        # Si la imagen es muy grande la reduce a un ancho de 2048 (para poder procesarla)
        if (result["original_size"]["width"] > 2048):
            img = self.image_resize(img, width=2048)
            cv2.imwrite(self.filename, img)

        # Si la imagen tiene menos de 640x480 (o uno de los dos) se rechaza por resolucion
        elif ((result["original_size"]["width"] < 640) or (result["original_size"]["height"] < 800)):
            motivos.append("Imagen pequena")
            result["status"] = "rechazado"
            
        # Si cumple las condiciones recorta la imagen alrededor de la cara
        if img is not None and result["status"] == "aprobado":        
            self.crop_image(self.filename, self.doc)
            import base64
            if (path.exists("crops/face_" + self.doc + ".jpg")):
                imgOut = cv2.imread("crops/crop_" + self.doc + ".jpg")
                result["final_size"] = self.test_size(imgOut)
                string = base64.b64encode(cv2.imencode('.jpg', imgOut)[1]).decode()
                result["image"] = string
                result["outfile"] = "crop_" + self.doc
            else:
                result["status"] = "rechazado"
                motivos.append("No fue posible crear recorte de imagen")
                result["image"] = ""

        # Si la imagen se ha rechazado , se actualiza los motivos
        if result["status"] != "aprobado":
            result["motivos"]= motivos

        if(DEBUG_JSON):
            with open('crops/response_' + self.doc + '.json', 'w') as json_file:
                json.dump(result, json_file)

        result["process_time"] = time.time() - start
        
        # Se retorna el JSON del resultado
        return result



# ip = Imp('uploads/face1.jpg', 'imagen1')
# print(ip)

# result = ip.test_image()

# if 'image' in result:
#     del result['image']
# print(result)
