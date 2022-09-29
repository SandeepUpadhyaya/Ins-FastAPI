some working codes to set up mlflow
tracking server setting
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

backend server/db setting
mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root .\mlruns