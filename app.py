import os
import time
import re
import fitz  # PyMuPDF
import requests
import sqlite3
import shutil
from difflib import SequenceMatcher
from urllib.parse import quote, unquote
from flask import Flask, request, render_template, send_from_directory, jsonify
from PIL import Image
import pytesseract
import io
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread


app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["20 per minute"])

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
document_folder = os.path.join(BASE_DIR, "downloaded_docs")

HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = os.getenv("HF_API_KEY") or "YOUR_HUGGINGFACE_API_KEY_HERE"


document_cache = {}



tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"[INFO] Tesseract found at: {tesseract_path}")
else:
    fallback_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(fallback_win_path):
        pytesseract.pytesseract.tesseract_cmd = fallback_win_path
        print(f"[INFO] Tesseract fallback path used: {fallback_win_path}")
    else:
        print("[ERROR] Tesseract not found. OCR will fail unless installed.")

@app.route("/api/add-document", methods=["POST"])
def add_document():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Please upload a PDF file."}), 400

    save_path = os.path.join(document_folder, uploaded_file.filename)
    uploaded_file.save(save_path)

    content = extract_text_from_pdf(save_path)
    normalized = normalize(content)
    conn = sqlite3.connect("docs.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO documents (filename, path, normalized_text) VALUES (?, ?, ?)", 
              (uploaded_file.filename, save_path, normalized))
    conn.commit()
    conn.close()

    document_cache[uploaded_file.filename] = {
        "text": normalized,
        "path": save_path
    }

    return jsonify({"success": True, "message": "New document successfully added!", "filename": uploaded_file.filename})

def extract_text_with_images(doc):
    text = ""
    for page in doc:
        text += page.get_text()  

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            text += "\n" + pytesseract.image_to_string(img)
    return text

def init_db():
    conn = sqlite3.connect("docs.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            filename TEXT PRIMARY KEY,
            path TEXT,
            normalized_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def normalize(text):
    return re.sub(r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]', '', text.lower()).strip()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_text_from_pdf(path, max_pages=9):
    try:
        doc = fitz.open(path)
        text = extract_text_with_images(doc)
        doc.close()
        return text
    except Exception as e:
        print(f"[ERROR] Failed to extract text: {e}")
        return ""


def get_documents():
    return {
        file: os.path.join(document_folder, file)
        for file in os.listdir(document_folder)
        if file.lower().endswith(".pdf")
    }

def preload_documents():
    import sqlite3
    conn = sqlite3.connect("docs.db")
    c = conn.cursor()

    c.execute("SELECT filename, path, normalized_text FROM documents")
    rows = c.fetchall()
    for filename, path, normalized in rows:
        document_cache[filename] = {"text": normalized, "path": path}
    print(f"[INFO] preload_documents: {len(rows)} file loaded from DB.")

    files_in_folder = get_documents()
    for filename, path in files_in_folder.items():
        if filename in document_cache:
            continue  

        print(f"[INFO] New file found, OCR starts: {filename}")
        content = extract_text_from_pdf(path)
        normalized = normalize(content)

        document_cache[filename] = {"text": normalized, "path": path}
        c.execute("INSERT OR REPLACE INTO documents (filename, path, normalized_text) VALUES (?, ?, ?)", 
                  (filename, path, normalized))

    conn.commit()
    conn.close()



def find_best_match_with_ai(query, options):
    prompt = f"User search: '{query}'\n\nSelect the most relevant of the filenames below and return only the filename:\n"
    for i, opt in enumerate(options):
        prompt += f"{i+1}. {opt}\n"
    prompt += "\nThe answer should only be the file name:"

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt})
    result = response.json()

    if isinstance(result, list) and "generated_text" in result[0]:
        output = result[0]["generated_text"].strip().split("\n")[-1]
        for opt in options:
            if output.strip() in opt:
                return opt
    return None

def find_matching_document(query):
    keywords = normalize(query).split()
    best_match = None
    best_score = 0

    for name, data in document_cache.items():
        text = data["text"]
        chunks = text.split()[:500]
        
        scores = []
        for word in keywords:
            if chunks:
                score = max([similarity(word, chunk) for chunk in chunks])
                scores.append(score)
            else:
                scores.append(0)

        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score > best_score:
            best_score = avg_score
            best_match = name

    if best_score > 0.55:
        return [f"/download/{quote(best_match)}"]

    ai_choice = find_best_match_with_ai(query, list(document_cache.keys()))
    if ai_choice:
        return [f"/download/{quote(ai_choice)}"]

    return None


def ai_chatbot_response(query):
    blocked_words = ['hack', 'sql', 'curl', 'rm -rf', 'exploit']
    if any(word in query.lower() for word in blocked_words):
        return "Inappropriate content detected."

    links = find_matching_document(query)
    if links:
        response = ""
        filename_main = unquote(links[0].split('/')[-1])
        response += f"📄 Relevant Document: <a href='{links[0]}'>{filename_main}</a>"
        if len(links) > 1:
            filename_alt = unquote(links[1].split('/')[-1])
            response += f"<br>💡 Alternative: <a href='{links[1]}'>{filename_alt}</a>"
        return response
    else:
        return '''
        ❌ No suitable document found. Please try again with a different word.
        <br><br>
        <form method="POST">
            <input type="text" name="query" placeholder="Do you want to ask anything else?" required style="width: 80%; padding: 10px;">
            <button type="submit">🔎 Sor</button>
        </form>
        '''

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            return render_template("index.html", query=None, ai_response=None, chatbot_visible=True)
        ai_response = ai_chatbot_response(query)
        return render_template("index.html", query=query, ai_response=ai_response, chatbot_visible=True)
    return render_template("index.html", query=None, ai_response=None, chatbot_visible=False)

@app.route("/api/query", methods=["POST"])
def query_api():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"response": "⚠️ Please enter a query."})
    response = ai_chatbot_response(query)
    return jsonify({"response": response})

@app.route("/download/<path:filename>")
def download_file(filename):
    decoded_filename = unquote(filename)
    return send_from_directory(document_folder, decoded_filename, as_attachment=True)

@app.route("/refresh")
def refresh_docs():
    preload_documents()
    return "Documents successfully rescanned! "

def start_file_watcher():
    class PDFHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and event.src_path.lower().endswith(".pdf"):
                filename = os.path.basename(event.src_path)
                print(f"[WATCHDOG] New PDF detected: {filename}")

                content = extract_text_from_pdf(event.src_path)
                normalized = normalize(content)

                conn = sqlite3.connect("docs.db")
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO documents (filename, path, normalized_text) VALUES (?, ?, ?)",
                        (filename, event.src_path, normalized))
                conn.commit()
                conn.close()

                document_cache[filename] = {
                    "text": normalized,
                    "path": event.src_path
                }

                print(f"[WATCHDOG] {filename} added to cache and DB.")

    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, path=document_folder, recursive=False)
    observer.start()
    print(f"[WATCHDOG] Watching folder: {document_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    init_db()
    preload_documents()
    Thread(target=start_file_watcher, daemon=True).start()
    app.run(debug=True)
    

