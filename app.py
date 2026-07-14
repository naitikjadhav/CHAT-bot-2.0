import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from chatbot_engine import CollegeChatbot

# Initialize Flask application and Chatbot
app = Flask(__name__)
bot = CollegeChatbot(intents_path="intents.json")

# Document Upload Configurations
UPLOAD_FOLDER = 'uploaded_documents'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB size limit

# Ensure the target upload folder exists directory-wise
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- USER CHATBOT ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")
    result = bot.get_response(user_message)
    return jsonify(result)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"response": "No file part in the request.", "success": False}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"response": "No file selected.", "success": False}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({
            "response": f"Successfully uploaded '{filename}'! The administration team will verify it shortly.",
            "success": True
        })
    
    return jsonify({"response": "Invalid file type. Please upload a PDF, DOCX, or Image.", "success": False}), 400

# --- ADMIN PANEL ROUTES ---

@app.route("/admin")
def admin_panel():
    # Read the upload directory to get a list of all current files
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
    except FileNotFoundError:
        files = []
    return render_template("admin.html", files=files)

@app.route("/uploaded_documents/<filename>")
def view_file(filename):
    # Safely serve files from the directory using secure Flask protocols
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    # Render binds dynamic production ports via the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    # debug=False prevents system profile exploits in a live environment
    app.run(host="0.0.0.0", port=port, debug=False)
