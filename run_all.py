import subprocess
import os
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Utility to run shell commands and handle errors."""
    print(f"\n{'='*60}")
    print(f"[*] STARTING: {description}")
    print(f"{'='*60}")
    
    try:
        # We use sys.executable to ensure we use the same python environment
        process = subprocess.Popen([sys.executable] + command.split(), 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT,
                                   text=True)
        
        # Stream output to console
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[!] ERROR: {description} failed with return code {process.returncode}")
            return False
            
        print(f"\n[+] COMPLETED: {description}")
        return True
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {str(e)}")
        return False

def main():
    # 1. Ensure project structure exists
    directories = ['checkpoints', 'data', 'outputs', 'mlruns']
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)

    start_total = time.time()

    # 2. Execution Sequence
    steps = [
        ("train_ddpm.py", "Training Pixel-space DDPM Model"),
        ("train_sd.py", "Training Latent-space Stable Diffusion Model"),
        ("compare_inference.py", "Running I2I Comparison Benchmark")
    ]

    for script, desc in steps:
        if not os.path.exists(script):
            print(f"[!] Target script {script} not found. Skipping...")
            continue
            
        success = run_command(script, desc)
        if not success:
            print("\n[!] Pipeline halted due to error.")
            sys.exit(1)

    total_duration = (time.time() - start_total) / 60
    print(f"\n{'='*60}")
    print(f"ALL STEPS COMPLETED SUCCESSFULLY")
    print(f"Total Pipeline Time: {total_duration:.2f} minutes")
    print(f"View results by running: mlflow ui")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()