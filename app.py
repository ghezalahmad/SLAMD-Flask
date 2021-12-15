from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import pandas as pd
import csv

app = Flask(__name__)
socketio = SocketIO(app)


# textarea that accept csv format text
@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        results = []
        user_csv = request.form.get('user_csv').split('\n')
        reader = csv.DictReader(user_csv)
        for row in reader:
            results.append(dict(row))
        fieldnames = [key for key in results[0].keys()]
        return render_template('home.html', results=results, fieldnames=fieldnames, len=len)






# uploading file 
@app.route('/upload',methods = ['GET','POST'])
def upload_route_summary():
    if request.method == 'POST':
            # Create variable for uploaded file
        file = request.files['file']
        file.save(os.path.join("uploaded", file.filename))
        return render_template('upload.html', message='uploaded successfully')
    return render_template('upload.html', message='upload')




# Invalid url
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404







#df = pd.read_csv('uploaded/Simp_Embedded_Data_28d_cubic_strength.csv')
#print(df.head())
if __name__ == '__main__':
    import os
    HOST = os.environ.get("SERVER_HOST", 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT=5555
    app.run(debug=True)
