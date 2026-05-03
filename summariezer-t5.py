from flask import Flask, render_template, request, redirect, url_for
import os
import PyPDF2
from werkzeug.utils import secure_filename
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    print("Extracted text:", text)  # Add this print statement
    return text

def summarize_text(text):
    print("Input text:", text)  # Add this print statement
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0])
    print("Summary:", summary)  # Add this print statement
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return redirect(request.url)
    if pdf_file and allowed_file(pdf_file.filename):
        filename = secure_filename(pdf_file.filename)
        pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        summary = summarize_text(text)
        return render_template('result.html', text=text, summary=summary)
    return "Error uploading file"

if __name__ == '__main__':
    app.run(debug=True)
