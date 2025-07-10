from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from rag.embedder import MedicalEmbedder
from rag.retriever import Retriever
from rag.prompt import build_prompt
from llm.model import ask_llm

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize once
embedder = MedicalEmbedder("data/")
retriever = Retriever(embedder=embedder)

@app.route("/", methods=["GET"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("chat.html", chat_history=session["chat_history"])

@app.route("/get", methods=["POST"])
def chat():
    question = request.form.get("question")
    if not question:
        return redirect(url_for("index"))

    contexts = retriever.search(question)
    prompt = build_prompt(contexts, question)
    answer = ask_llm(prompt)

    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"question": question, "answer": answer})
    session.modified = True

    return render_template("chat.html", chat_history=session["chat_history"])

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

# ✅ Auto-Triage Agent Endpoint
@app.route("/triage", methods=["POST"])
def triage():
    data = request.get_json()
    user_input = data.get("input", "")
    followup = embedder.detect_symptoms_and_followup(user_input)
    return jsonify({"followup": followup or "No symptoms detected."})

# ✅ Multimodal Medical Report Analyzer Endpoint
@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_uploads", filename)
    os.makedirs("temp_uploads", exist_ok=True)
    file.save(filepath)

    summary = embedder.analyze_medical_image(filepath)
    os.remove(filepath)

    # Add to chat history
    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({
        "question": "🧾 Uploaded medical report",
        "answer": summary
    })
    session.modified = True

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="10.20.122.42", port=8080, debug=True)
