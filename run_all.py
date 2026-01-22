import os
import sys

def run_command(command, stage_name):
    print(f"\n{'='*40}")
    print(f"[*] STARTING: {stage_name}")
    print(f"{'='*40}")
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"[!] ERROR: {stage_name} failed!")
        sys.exit(1)
    print(f"[+] COMPLETED: {stage_name}")

def main():
    # 1. Train Pixel Model
    run_command("python train_ddpm.py", "Training DDPM (Pixel Space)")
    
    # 2. Train Latent Model
    run_command("python train_sd.py", "Training Stable Diffusion (Latent Space)")
    
    # 3. Run Benchmark
    run_command("python compare_inference.py", "Running I2I Benchmark")

if __name__ == "__main__":
    main()
