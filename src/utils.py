import torch
import yaml
import mlflow
import time
from functools import wraps

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def setup_mlflow(config):
    mlflow.set_experiment(config['mlflow']['experiment_name'])

def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        print(f"[*] {func.__name__} took {duration:.2f} seconds")
        # Log to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_metric(f"time_{func.__name__}", duration)
        return result
    return wrapper
