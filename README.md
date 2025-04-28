Document Chatbot

Intelligent document search system - finds the most relevant documents based on the user's query and provides them as downloadable links.  
Reads PDF content, matches meaning with artificial intelligence and provides alternative directions.

---

## Features

- Matches content by reading the first pages of PDF files  
- Matches filename with AI via Hugging Face if necessary  
- Provides 1 most relevant main document and 1 alternative document  
- Rate limit & harmful query filtering  
- Modern, dark-mode supported HTML interface  
- Also available via API (`/api/query`)

---

## Tesseract Installation (Windows)

1. [Download Tesseract OCR](https://github.com/tesseract-ocr/tesseract/releases)
2. Check **"Add to PATH ”** during installation
3. Alternatively, you can install it in this folder:

##  Setup

```bash
git clone https://github.com/kullaniciadi/smart-doc-chatbot.git
cd smart-doc-chatbot
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
