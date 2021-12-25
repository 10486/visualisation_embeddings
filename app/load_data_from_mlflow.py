from mlflow.tracking import MlflowClient
from pathlib import Path
import pandas as pd
import mlflow
import shutil

def download_mlflow_artifacts(experiment_name, artifact_paths):
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    df = mlflow.search_runs([experiment_id])
    data = pd.DataFrame({
        "run_id": df["run_id"],
        "end_date": df["end_time"].dt.date,
        "end_time": df["end_time"].dt.strftime("%H:%M")
    })
    data = data.loc[:4,:]

    path = Path(f"./app/data/")
    print("Выгрузка данных с 5 последних экспериментов")
    for run_id in data["run_id"]:
        download_run_artifacts(client, run_id, path, artifact_paths)

def download_run_artifacts(client, run_id, path, artifact_paths):
    path = path / run_id
    path.mkdir(exist_ok=True, parents=True)
    for artifact in artifact_paths:
        try:
            client.download_artifacts(run_id, artifact, path)
        except:
            print(f"Артефакта '{artifact}' нет в эксперименте с run id '{run_id}'")



if __name__ == '__main__':
    artifact_paths = ["distance analyze"]
    experiment_name = "Skolkovo_4"
    # добавить удаление папки data
    download_mlflow_artifacts(experiment_name, artifact_paths)
