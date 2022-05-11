from fastai.tabular.all import *
import pathlib
import os

#api imports
from flask import Flask, request, render_template

#model preparation

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

path = Path(os.getcwd())
full_path = os.path.join(path,'export.pkl')

model = load_learner(full_path)

#api configuration

API = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

API.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@API.route("/", methods=["GET"])
def serverOn():
  return render_template('index.html')

@API.route("/model", methods=["POST"])
def useModel():
  if request.method == 'POST':
    if 'file' not in request.files:
      return 'There is no file in request'

    file = request.files['file']
    if file.filename == '':
        return 'Invalid file name'
    if file and allowed_file(file.filename):
      file_ext = file.filename.rsplit('.', 1)[1].lower()
      file.save(os.path.join(API.config['UPLOAD_FOLDER'], f'modelo.{file_ext}'))
      file_path = os.path.join(path,f'uploads', f'modelo.{file_ext}')
      pred = model.predict(file_path)
      img_dict = {
          'Vinsmoke Sanji': 'Vinsmoke Sanji',
          'Roronoa Zoro': 'Roronoa Zoro',
          'Monkey D. Luffy': 'Monkey D. Luffy'
      }
      return img_dict.get(pred[0], '')
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
  print("Start")
  port = int(os.environ.get("PORT", 5000))
  API.run(host='0.0.0.0', port=port)