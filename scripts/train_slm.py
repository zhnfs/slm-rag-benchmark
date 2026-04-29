import subprocess
import os
import sys

def run_training():
    # Paths
    benchmark_dir = "/Users/hainingzheng/pythonCodes/slm-rag-benchmark"
    factory_dir = "/Users/hainingzheng/pythonCodes/finetune_slm_llamaFactory"
    config_path = os.path.join(factory_dir, "examples/train_lora/rag_finetune_qwen2_5_0_5b.yaml")
    
    print("🚀 Initializing SLM Behavioral Fine-Tuning...")
    print(f"📍 Benchmark Repo: {benchmark_dir}")
    print(f"📍 LLaMA-Factory: {factory_dir}")
    
    # 1. Locate the llamafactory-cli executable in the factory venv
    cli_path = os.path.join(factory_dir, ".venv/bin/llamafactory-cli")
    
    if not os.path.exists(cli_path):
        # Fallback to just the command if venv isn't found
        cli_path = "llamafactory-cli"
        print("⚠️  Warning: .venv/bin/llamafactory-cli not found, falling back to global command.")

    # 2. Ensure we are in the LLaMA-Factory directory to run the command
    os.chdir(factory_dir)
    
    # 3. Construct the training command
    cmd = [
        cli_path, "train",
        config_path
    ]
    
    print(f"🏃 Running command: {' '.join(cmd)}")
    
    try:
        # 3. Execute training
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output to console
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Training Complete! LoRA weights saved to: saves/qwen2.5-0.5b/lora/rag_v1")
        else:
            print(f"\n❌ Training failed with exit code {process.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error launching training: {e}")

if __name__ == "__main__":
    run_training()
