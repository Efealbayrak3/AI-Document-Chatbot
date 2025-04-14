import os
import re
import fitz  # PyMuPDF
import requests
from difflib import SequenceMatcher
from urllib.parse import quote, unquote
from flask import Flask, request, render_template, send_from_directory, jsonify

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
document_folder = os.path.join(BASE_DIR, "downloaded_docs")

HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = os.getenv("HF_API_KEY") or "YOUR_HUGGINGFACE_API_KEY_HERE"


document_cache = {}

def normalize(text):
    return re.sub(r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]', '', text.lower()).strip()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_text_from_pdf(path, max_pages=3):
    try:
        doc = fitz.open(path)
        text = ''
        for page in doc[:max_pages]:
            text += page.get_text()
        doc.close()
        return text
    except Exception:
        return ""

def get_documents():
    return {
        file: os.path.join(document_folder, file)
        for file in os.listdir(document_folder)
        if file.lower().endswith(".pdf")
    }

def preload_documents():
    global document_cache
    document_cache = {}
    documents = get_documents()
    for filename, path in documents.items():
        content = extract_text_from_pdf(path)
        document_cache[filename] = {
            "text": normalize(content),
            "path": path
        }
    print(f"[INFO] preload_documents: {len(document_cache)} file cached.")

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
        scores = [max([similarity(word, chunk) for chunk in text.split()[:500]]) for word in keywords]
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
    return "📂 Documents successfully rescanned! "

if __name__ == "__main__":
    preload_documents()
    app.run(debug=True)
