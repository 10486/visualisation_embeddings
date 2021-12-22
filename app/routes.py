from app import app
from flask import render_template,request,jsonify
from app.read_and_prepare_2d_dots import read_embeddings, get_all_dots
from pathlib import Path

@app.route("/")
@app.route("/<perplexity>")
def index(perplexity=5):
    path = Path("./app/data")
    embeddings = get_all_dots(path,perplexity=int(perplexity))
    # тексты которктих стори
    short = [f"text# {x}" for x in range(len(embeddings.short[0]))]
    # тексты длинных стори
    long = [f"text# {x}" for x in range(len(embeddings.long[0]))]
    return render_template("index.html", embeddings=embeddings, short=short, long=long)
