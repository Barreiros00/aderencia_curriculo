from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Certifique-se de que o diretório de uploads existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def find_non_matching_words(curriculo_text, vaga_text):
    curriculo_words = set(curriculo_text.split())
    vaga_words = set(vaga_text.split())
    non_matching_words = vaga_words - curriculo_words
    return non_matching_words

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'curriculo' not in request.files or 'vaga_text' not in request.form:
            return redirect(request.url)

        curriculo = request.files['curriculo']
        vaga_text = request.form['vaga_text']

        if curriculo.filename == '' or vaga_text.strip() == '':
            return redirect(request.url)

        if curriculo and allowed_file(curriculo.filename):
            curriculo_filename = secure_filename(curriculo.filename)
            curriculo_path = os.path.join(app.config['UPLOAD_FOLDER'], curriculo_filename)
            curriculo.save(curriculo_path)

            curriculo_text = extract_text_from_pdf(curriculo_path)
            curriculo_text = preprocess_text(curriculo_text)
            vaga_text = preprocess_text(vaga_text)

            similarity = calculate_similarity(curriculo_text, vaga_text)
            adherence_percentage = round(similarity * 100, 2)
            non_matching_words = find_non_matching_words(curriculo_text, vaga_text)

            return render_template('index.html', adherence=adherence_percentage, non_matching_words=non_matching_words)

    return render_template('index.html', adherence=None, non_matching_words=None)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
