from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3, os
from PyPDF2 import PdfReader

# phi imports
from phi.agent import Agent, RunResponse
from phi.model.google import Gemini

app = Flask(__name__)
CORS(app)

S3_BUCKET = 'comparator'
s3 = boto3.client('s3')

UPLOAD_COUNTER_FILE = "upload_count.txt"
uploaded_texts = {}

def get_next_filename():
    if not os.path.exists(UPLOAD_COUNTER_FILE):
        with open(UPLOAD_COUNTER_FILE, "w") as f:
            f.write("0")

    with open(UPLOAD_COUNTER_FILE, "r") as f:
        count = int(f.read().strip())

    count += 1
    with open(UPLOAD_COUNTER_FILE, "w") as f:
        f.write(str(count))

    return f"file{count}.pdf"

rag_agent = Agent(model=Gemini(id="gemini-2.5-flash-preview-05-20"))

@app.route("/")
def home():
    return "RAG backend with Chatbot is running!"

@app.route("/upload_rag", methods=["POST"])
def upload_file():
    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "Missing file"}), 400

        filename = get_next_filename()
        file.save(filename)

        s3.upload_file(filename, S3_BUCKET, filename)
        s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{filename}"

        reader = PdfReader(filename)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        uploaded_texts[filename] = text

        os.remove(filename)

        return jsonify({
            "message": f"Uploaded and ingested {filename}",
            "s3_url": s3_url,
            "total_pdfs": len(uploaded_texts)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rag_query', methods=['POST'])
def handle_rag_query():
    try:
        data = request.json
        print("Incoming JSON:", request.json)
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query parameter is missing."}), 400

        all_text = "\n".join(uploaded_texts.values())
        if not all_text.strip():
            return jsonify({"error": "No documents uploaded yet."}), 400

        prompt = f"""
        You are a helpful assistant. Use the following context to answer the query.

        Context:
        {all_text}

        Query: {query}
        """

        run: RunResponse = rag_agent.run(prompt)
        return jsonify({"response": run.content}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query parameter is missing."}), 400

        prompt = f"You are a helpful chatbot. Answer the following:\n\n{query}"
        run: RunResponse = rag_agent.run(prompt)

        return jsonify({"response": run.content}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


