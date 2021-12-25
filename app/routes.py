from app.read_and_prepare_2d_dots import get_all_dots
from flask import render_template
from pathlib import Path
from app import app

@app.route("/")
@app.route("/<int:perplexity>")
def index(perplexity=5):
    path = Path("./app/data")
    embeddings = get_all_dots(path,perplexity=int(perplexity))
    # тексты которктих стори
    short = [f"text# {x}" for x in range(len(embeddings.short[0]))]
    # тексты длинных стори
    long = [f"text# {x}" for x in range(len(embeddings.long[0]))]
    return render_template("index.html", embeddings=embeddings, short=short, long=long, perplexity=perplexity)
