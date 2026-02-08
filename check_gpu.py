"""Quick check for why TensorFlow might not see your NVIDIA GPU."""
import subprocess
import sys

def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    except Exception as e:
        return type("R", (), {"returncode": -1, "stdout": "", "stderr": str(e)})()

print("=== TensorFlow ===")
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Check if CPU-only package (often has "+cpu" in version or different package name)
try:
    import pkg_resources
    dist = pkg_resources.get_distribution("tensorflow")
    print(f"Package: {dist.project_name} {dist.version}")
    if "cpu" in dist.project_name.lower() or "+cpu" in dist.version.lower():
        print("  >>> You have a CPU-only build. Uninstall and install 'tensorflow' (GPU support).")
except Exception:
    pass

print("\n=== GPU detection ===")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPUs found: {gpus}")
else:
    print("No GPUs reported by TensorFlow.")

print("\n=== NVIDIA driver (nvidia-smi) ===")
r = run("nvidia-smi")
if r.returncode == 0:
    print(r.stdout.strip() or "(no output)")
else:
    print("nvidia-smi failed or not found. Install/update NVIDIA driver.")

print("\n=== Common fixes if no GPU ===")
print("1. Install cuDNN (TensorFlow needs BOTH CUDA and cuDNN):")
print("   https://developer.nvidia.com/cudnn - match your CUDA version.")
print("2. Match versions: TF 2.15 needs CUDA 12.2 + cuDNN 8.9.")
print("3. Add CUDA and cuDNN bin folders to PATH (e.g. C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin).")
print("4. Restart terminal (or reboot) after installing CUDA/cuDNN.")
print("5. If using conda: conda install cuda-toolkit cudnn")
