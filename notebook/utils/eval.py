import utils.load
import torch
import time
import numpy as np
import pandas as pd

from typing import List, Dict, Any
from sklearn import metrics as sk_metrics

def get_size(model: torch.nn.Module) -> float:
    """
    Calculates the total memory footprint of a PyTorch model in Megabytes.

    This function sums the memory occupied by both trainable parameters 
    (weights/biases) and non-trainable buffers (e.g., BatchNorm running means).

    Args:
        model (torch.nn.Module): The PyTorch model to measure.

    Returns:
        float: The total size of the model in Megabytes (MB).
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_in_mb = (param_size + buffer_size) / (1024**2)
    return size_in_mb


def measure_time(model: torch.nn.Module, 
                 input_data: torch.Tensor, 
                 num_iterations: int = 100) -> float:
    """
    Measures the average inference latency of a model over multiple iterations.

    Includes a warmup phase to initialize the CUDA context/cache and 
    uses synchronization to ensure accurate timing on GPU devices.

    Args:
        model (torch.nn.Module): The model to benchmark.
        input_data (torch.Tensor): A sample input batch (e.g., torch.randn(1, 3, 224, 224)).
        num_iterations (int): Number of times to run the forward pass for averaging.

    Returns:
        float: The average inference time per iteration in milliseconds (ms).
    """
    model.eval()
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # Calculate average time in milliseconds
    avg_time_ms = ((end_time - start_time) / num_iterations) * 1000
    return avg_time_ms

def test_model_metrics(model, test_loader, device, selected_metrics=None, task='classification'):
    """
    Evaluates a model with user-selected metrics.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader
        device: 'cuda' or 'cpu'
        selected_metrics: List of strings like ['accuracy', 'precision', 'mse', 'mae']
        task: 'classification' or 'regression'
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = [] 

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if task == 'classification':
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            else:
                all_preds.extend(outputs.cpu().numpy())
                
            all_targets.extend(targets.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Dictionary mapping strings to scikit-learn functions
    metric_map = {
        # Classification
        'accuracy':  lambda: sk_metrics.accuracy_score(y_true, y_pred),
        'precision': lambda: sk_metrics.precision_score(y_true, y_pred, average='macro'),
        'recall':    lambda: sk_metrics.recall_score(y_true, y_pred, average='macro'),
        'f1':        lambda: sk_metrics.f1_score(y_true, y_pred, average='macro'),
        'auc':       lambda: sk_metrics.roc_auc_score(y_true, y_prob, multi_class='ovr'),
        # Regression
        'mae':       lambda: sk_metrics.mean_absolute_error(y_true, y_pred),
        'mse':       lambda: sk_metrics.mean_squared_error(y_true, y_pred),
        'rmse':      lambda: np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)),
        'r2':        lambda: sk_metrics.r2_score(y_true, y_pred)
    }

    # If no metrics chosen, provide defaults based on task
    if selected_metrics is None:
        selected_metrics = ['accuracy', 'f1'] if task == 'classification' else ['mae', 'rmse']

    results = {}
    for m in selected_metrics:
        if m in metric_map:
            results[m] = metric_map[m]()
        else:
            print(f"Warning: Metric '{m}' is not recognized.")

    return results

def evaluate_efficiency(model: torch.nn.Module, 
                        test_loader: torch.utils.data.DataLoader, 
                        metrics: List[str], 
                        task: str = 'classification', 
                        device: str = "cuda") -> Dict[str, Any]:
    """
    Comprehensive profiler that calculates model size, inference speed, 
    and predictive performance metrics in a single call.
    """
    model.to(device)
    model.eval()

    # 1. Calculate Model Size (MB)
    model_size = get_size(model)
    
    # 2. Measure Inference Time (Latency)
    try:
        data_iter = iter(test_loader)
        batch = next(data_iter)
        inputs = batch[0] 
        sample = inputs[0:1].to(device) 
        inf_time = measure_time(model, sample)
    except StopIteration:
        print("Error: test_loader is empty.")
        inf_time = 0.0

    # 3. Evaluate Predictive Metrics (Accuracy, F1, etc.)
    test_results = test_model_metrics(
        model=model, 
        test_loader=test_loader, 
        device=device, 
        selected_metrics=metrics, 
        task=task
    )
    
    # 4. Consolidate Results
    return {
        "metrics": test_results,
        "model_size_mb": round(model_size, 2),
        "inference_time_ms": round(inf_time, 4)
    }
    

def compare(model_path, test_loader, metrics):
    """
    Evaluates all models in a directory and returns a sorted Pandas DataFrame 
    comparing their size, speed, and accuracy metrics.
    """
    model_map = utils.load.load_all_models(model_path, device="cuda")
    results_list = []

    for name, model in model_map.items():
        stats = evaluate_efficiency(model, test_loader, metrics, task='classification')
        
        row = {
            "Model Name": name,
            "Size (MB)": stats['model_size_mb'],
            "Latency (ms)": stats['inference_time_ms']
        }
        
        # Flatten the metrics dictionary into the row
        row.update(stats['metrics'])
        
        results_list.append(row)
        
        print(f"\nModel: {name}")
        print(f"  - Size: {stats['model_size_mb']} MB")
        print(f"  - Latency: {stats['inference_time_ms']} ms")
        print(f"  - Performance: {stats['metrics']}")
        
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results_list)
    
    if metrics:
        df = df.sort_values(by=metrics[0], ascending=False)
        
    return df