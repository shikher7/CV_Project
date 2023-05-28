from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # replace with your secret key

users = {}  # This dictionary will act as a simple database

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        phone = request.form.get('phone')
        if email in users:
            return 'User already exists'
        else:
            users[email] = {
                "password": generate_password_hash(password),
                "name": name,
                "phone": phone
            }
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users and check_password_hash(users[email]["password"], password):
            session['user'] = email
            session['name'] = users[email]["name"]
            session['phone'] = users[email]["phone"]
            return redirect(url_for('form'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'user' in session:
        return render_template('form.html', name=session['name'], phone=session['phone'])
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
