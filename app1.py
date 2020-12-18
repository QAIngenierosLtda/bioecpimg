from flask import Flask, request, send_file
from PIL import Image
import base64



app = Flask(__name__)

@app.route('/')
def index():
  return 'Server Works!'

@app.route('/greet')
def say_hello():
  return 'Hello world'

@app.route('/img_test', methods=['GET','POST'])
def img_test():
  # image = request.args.get('img', default = '', type = str)
#   size = request.form['size']
  #login(arg,arg) is a function that tries to log in and returns true or false
  file = request.args.get('file')
  starter = file.find(',')
  image_data = file[starter+1:]
  image_data = bytes(image_data, encoding="ascii")
  with open('.../image.jpg', 'wb') as fh:
      fh.write(base64.decodebytes(image_data))
  return 'ok'

@app.route('/get_image')
def get_image():
    if request.args.get('type') == '1':
       filename = 'ok.png'
    else:
       filename = 'error.png'
    return send_file(filename, mimetype='image/gif')


if __name__ == '__main__':
    app.run(debug=True)