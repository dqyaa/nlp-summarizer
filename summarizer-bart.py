from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from transformers import BartForConditionalGeneration, BartTokenizer
import PyPDF2
import os

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
    text = ' '.join(text.split())
    return text

def summarize_text(text):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    print("Input tokens:", inputs['input_ids'])
    summary_ids = model.generate(inputs.input_ids, max_length=150000, min_length=400, length_penalty=2.0, num_beams=4, early_stopping=True)
    print("Summary tokens:", summary_ids)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Generated summary:", summary)
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
        return render_template('result.html', text=text, summary=summary)  # Pass summary to the template
    return "Error uploading file"

if __name__ == '__main__':
    app.run(debug=True)
