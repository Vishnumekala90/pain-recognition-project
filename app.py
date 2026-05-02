
from flask import Flask, render_template

import webbrowser
from threading import Timer
from admin.admin import admin_bp
from user.user import user_bp
import os
from utils.algorithms import get_algo

UPLOAD_FOLDER = 'uploads'
MAX_UPLOAD_SIZE_MB = 512  
app = Flask(__name__)

app.secret_key = "abcdesdfjsldjflksd" 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)




@app.route('/', methods=['GET', 'POST'])
def index():
    
    
    return render_template('react_index.html',title="Index")

@app.route('/project_info', methods=['GET', 'POST'])
def project_info():
    card_data = [
        {"title": "Existing Algorithm", "text": get_algo("existing")},
        {"title": "Proposed Algorithm", "text": get_algo("proposed")},
        {"title": "Extension Algorithm", "text": get_algo("extension")},
    ]
    return render_template('project_info.html', title="Project Info", card_data=card_data)
    
app.register_blueprint(admin_bp)
app.register_blueprint(user_bp)












def open_browser():
    webbrowser.open_new("http://localhost:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()

    app.run(debug=False)    
