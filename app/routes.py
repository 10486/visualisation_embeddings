from app.read_and_prepare_2d_dots import get_all_dots
from flask import render_template
from pathlib import Path
from app import app
import pandas as pd


@app.route("/")
def index():
    path = Path("./app/data")
    runs = [str(x.parts[-1]) for x in path.glob("*") if x.is_dir()]
    return render_template("index.html", runs=runs)

@app.route("/favicon.ico")
def favicon():
    return "нету"


@app.route("/<string:run_id>/")
@app.route("/<string:run_id>/<int:perplexity>")
def graph(run_id, perplexity=5):
    path = Path("./app/data")
    path = path / run_id

    embeddings_path = path / "embeddings"
    embeddings = get_all_dots(embeddings_path, perplexity=int(perplexity))

    texts_path = path / "sentences"
    long = pd.read_csv(texts_path/"db_sentences.csv")["stories"].str.replace("\n", " ").tolist()
    short = pd.read_csv(texts_path/"short_sentences.csv")["stories"].str.replace("\n", " ").tolist()

    data = {
        "embeddings":embeddings,
        "short":short,
        "long":long,
        "perplexity":perplexity,
        "run_id":run_id,
    }
    return render_template("graph.html", **data)
