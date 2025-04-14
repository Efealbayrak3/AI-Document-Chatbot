Document Chatbot

Akıllı döküman arama sistemi — kullanıcıdan gelen sorguya göre dökümanlar arasında en alakalı olanı bulur ve indirilebilir bağlantı olarak sunar.  
PDF içeriğini okur, yapay zeka ile anlam eşleşmesi yapar ve alternatif yönlendirmeler sağlar.

---

## Özellikler

-  PDF dosyalarının ilk sayfalarını okuyarak içerik eşleşmesi yapar  
-  Gerekirse Hugging Face üzerinden AI ile dosya adı eşleşmesi yapar  
-  En alakalı 1 ana dökümanı ve 1 alternatif dökümanı sunar  
-  Rate limit & zararlı sorgu filtreleme  
-  Modern, dark-mode destekli HTML arayüzü  
-  API üzerinden de kullanılabilir (`/api/query`)

---

##  Kurulum

```bash
git clone https://github.com/kullaniciadi/smart-doc-chatbot.git
cd smart-doc-chatbot
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
