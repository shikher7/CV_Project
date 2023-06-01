from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from face_detector import FaceDetector
from face_trainer import FaceTrainer
from face_recognizer import FaceRecognizer
import os
import json
import subprocess


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # replace with your secret key
dataset_path = 'dataset/'
trainer_path = 'trainer/'

# Load existing user database or create an empty one if it doesn't exist
try:
    with open('users_db.json', 'r') as f:
        users_db = json.load(f)
except FileNotFoundError:
    users_db = {}


@app.route('/face_login', methods=['GET','POST'])
def face_login():
    face_recognizer = FaceRecognizer(trainer_path)
    face_recognizer.recognize_face()
    user_id, confidence = str(face_recognizer.id), face_recognizer.confidence
    print(user_id, confidence)
    for user_id in users_db:
        print(user_id)
    # Check if the face is recognized with good confidence
    if confidence > 10:
        if user_id in users_db:
            session['user'] = session['name'] = users_db[user_id]["name"]
            return redirect(url_for('form'))
        else:
            return jsonify({'status': 'failure', 'message': 'User was not recognized.'})
    else:
        return jsonify({'status': 'failure', 'message': 'Face was not recognized.'})


@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'user' in session:
        subprocess.Popen(["python", "main.py"])  # runs main.py as a background process
        return render_template('form.html', name=session['name'])



@app.route('/', methods=['GET'])
def login():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        face_id = len(users_db) + 1  # assign the next available id

        # Check if user already exists
        if face_id in users_db:
            return 'User already exists'
        else:
            users_db[face_id] = {"name": name}  # add mapping from id to name
            os.makedirs(dataset_path + str(face_id))

            # Capture face and train
            FaceDetector(face_id).capture_face().close()
            FaceTrainer(dataset_path, trainer_path).train()

            # Update the user database
            with open('users_db.json', 'w') as f:
                json.dump(users_db, f)

            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('form'))


if __name__ == "__main__":
    app.run(debug=True)
