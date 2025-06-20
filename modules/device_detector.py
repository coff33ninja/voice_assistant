"""
Device Detector Script
Detects CPU/GPU availability, CUDA support, recommends appropriate CUDA/Torch version,
and logs system hardware info. Supports auto-download and setup of matching PyTorch builds.
Also detects AMD/Intel fallback and writes results to a .env file.
"""

import platform
import subprocess
import logging
import sys
from pathlib import Path

# Logger for this module
logger = logging.getLogger(__name__)

# Try importing GPUtil, install if not present
try:
    import GPUtil
except ImportError:
    logger.info("GPUtil not found. Attempting to install GPUtil...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "gputil"], check=True, capture_output=True, text=True)
        logger.info("GPUtil installed successfully.")
        import GPUtil
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install GPUtil. Command: '{' '.join(e.cmd)}'.")
        logger.error(f"Pip stdout: {e.stdout}")
        logger.error(f"Pip stderr: {e.stderr}")
        GPUtil = None # Ensure GPUtil is None if import or install fails
    except ImportError:
        logger.error("Failed to import GPUtil even after attempting installation.")
        GPUtil = None

# Dictionary mapping GPU name keywords to recommended CUDA/Torch versions
CUDA_COMPATIBILITY = {
    "RTX 30": {
        "cuda": "11.8",
        "torch": "2.1.0+cu118",
    },  # Supports Python 3.11, CUDA toolkit: https://developer.nvidia.com/cuda-11-8-0-download-archive
    "RTX 40": {
        "cuda": "12.8",
        "torch": "2.7.1+cu128",
    },  # Supports Python 3.11, CUDA toolkit: https://developer.nvidia.com/cuda-12-8-0-download-archive
    "RTX 50": {
        "cuda": "12.8",
        "torch": "2.7.1+cu128",
    },  # Supports Python 3.11, CUDA toolkit: https://developer.nvidia.com/cuda-12-8-0-download-archive
    "RTX 20": {
        "cuda": "11.3",
        "torch": "1.10.0+cu113",
    },  # Does not support Python 3.11; use Python 3.8, CUDA toolkit: https://developer.nvidia.com/cuda-11-3-0-download-archive
    "GTX 10": {
        "cuda": "10.2",
        "torch": "1.8.1+cu102",
    },  # Does not support Python 3.11; use Python 3.8, CUDA toolkit: https://developer.nvidia.com/cuda-10-2-download-archive
    "GTX 9": {
        "cuda": "10.1",
        "torch": "1.7.1+cu101",
    },  # Does not support Python 3.11; use Python 3.8, CUDA toolkit: https://developer.nvidia.com/cuda-10-1-download-archive
    "Quadro": {
        "cuda": "11.0",
        "torch": "1.9.1+cu110",
    },  # Does not support Python 3.11; use Python 3.8, CUDA toolkit: https://developer.nvidia.com/cuda-11-0-download-archive
    "Tesla": {
        "cuda": "11.4",
        "torch": "1.12.0+cu114",
    },  # Does not support Python 3.11; use Python 3.9, CUDA toolkit: https://developer.nvidia.com/cuda-11-4-0-download-archive
}

DEFAULT_CPU_TORCH = "2.2.2+cpu"  # For Python 3.11 compatibility


def get_cpu_info():
    return platform.processor() or platform.machine()


def get_gpu_info():
    if GPUtil is None:
        return ["GPUtil not available or failed to install. GPU detection skipped."]
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return ["No GPU detected by GPUtil."]
        return [f"{gpu.name} (ID {gpu.id}) - {gpu.memoryTotal}MB VRAM" for gpu in gpus]
    except Exception as e:
        return [f"Error detecting GPU via GPUtil: {str(e)}"]


def detect_cuda_with_torch():
    try:
        import torch
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available"):
            return torch.cuda.is_available()
        else:
            logger.warning("torch.cuda or torch.cuda.is_available not found in torch module.")
            return False
    except ImportError:
        logger.warning("PyTorch is not installed. CUDA availability check via PyTorch skipped.")
        return False
    except Exception as e:
        logger.error(f"Error during PyTorch CUDA check: {e}")
        return False


def get_cuda_device_name_with_torch():
    try:
        import torch
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "get_device_name"):
            if detect_cuda_with_torch():
                try:
                    return torch.cuda.get_device_name(0)
                except Exception as e:
                    logger.error(f"Error getting CUDA device name: {e}")
                    return "CUDA device name not accessible"
        return "CUDA not available or PyTorch issue"
    except ImportError:
        return "PyTorch not installed"
    except Exception as e:
        logger.error(f"Error in get_cuda_device_name_with_torch: {e}")
        return "CUDA not available or PyTorch issue"


def recommend_torch_version():
    # Always use torch 2.2.2 for Python 3.11 (latest supported, avoids 3.13 wheel issue)
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    if py_major == 3 and py_minor == 11:
        logger.info("Python 3.11 detected. Forcing torch version 2.2.2+cpu for compatibility.")
        return {"type": "cpu", "torch": "2.2.2+cpu"}

    try:
        import torch
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            if hasattr(torch.cuda, "get_device_name"):
                name = torch.cuda.get_device_name(0)
                for key, versions in CUDA_COMPATIBILITY.items():
                    if key.lower() in name.lower():
                        logger.info(f"GPU '{name}' matched with '{key}'. Recommending Torch: {versions['torch']}, CUDA: {versions['cuda']}")
                        return {"type": "cuda", "torch": versions['torch'], "cuda": versions['cuda']}
                logger.warning(f"CUDA is available but GPU '{name}' not found in CUDA_COMPATIBILITY map. Recommending a default CUDA-enabled PyTorch or CPU.")
                return {"type": "cuda", "torch": "2.0.1+cu118", "cuda": "11.8"} # Default to a common recent CUDA version
        # If torch.cuda is not available or no CUDA, fall through to CPU
    except ImportError:
        logger.info("PyTorch not imported, recommending CPU version.")
    except Exception as e:
        logger.error(f"Error during PyTorch version recommendation: {e}. Recommending CPU version.")

    return {"type": "cpu", "torch": DEFAULT_CPU_TORCH}


def detect_cpu_vendor():
    cpu_name = get_cpu_info().lower()
    if "amd" in cpu_name:
        return "amd"
    elif "intel" in cpu_name:
        return "intel"
    return "unknown"


def write_env_file(settings, base_dir_path, asr_device=None, tts_device=None):
    env_path = base_dir_path / ".env"
    logger.info(f"Writing hardware configuration to: {env_path}")
    # This overwrites the .env file. For more careful updates, python-dotenv's set_key could be used.
    with env_path.open("w") as f:
        for key, value in settings.items():
            f.write(f"{key.upper()}={value}\n")
        if asr_device:
            f.write(f"ASR_DEVICE={asr_device}\n")
            logger.info(f"ASR_DEVICE set to: {asr_device}")
        if tts_device:
            f.write(f"TTS_DEVICE={tts_device}\n")
            logger.info(f"TTS_DEVICE set to: {tts_device}")


def install_pytorch_version(torch_version_str):
    logger.info(f"Attempting to install PyTorch version: {torch_version_str}")

    if "+cu" in torch_version_str:
        cuda_suffix = torch_version_str.split('+')[-1]
        index_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
    else: # Assumes CPU
        index_url = "https://download.pytorch.org/whl/cpu"

    cmd = [sys.executable, "-m", "pip", "install", f"torch=={torch_version_str}", "torchvision", "torchaudio", "--index-url", index_url]
    logger.info(f"Executing PyTorch installation command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully installed PyTorch version: {torch_version_str} and associated libraries.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install specified PyTorch version. Command: '{' '.join(e.cmd)}'.")
        logger.error(f"Pip stdout: {e.stdout}")
        logger.error(f"Pip stderr: {e.stderr}")
        logger.error("Please check the PyTorch installation command and your internet connection. You may need to install PyTorch manually.")
        raise # Re-raise the exception to indicate failure of this step


def run_device_setup(base_dir_path_str):
    base_dir = Path(base_dir_path_str)
    logger.info("--- Running Hardware Detection and PyTorch Setup ---")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    cpu_info = get_cpu_info()
    logger.info(f"CPU: {cpu_info}")

    for gpu_info_line in get_gpu_info():
        logger.info(f"GPU: {gpu_info_line}")

    cuda_available_torch = detect_cuda_with_torch()
    logger.info(f"CUDA Available (according to PyTorch, if installed): {cuda_available_torch}")

    cuda_device_name = get_cuda_device_name_with_torch()
    logger.info(f"CUDA Device Name (from PyTorch): {cuda_device_name}")

    recommended_config = recommend_torch_version()
    logger.info(f"Recommended PyTorch: {recommended_config['torch']}")
    logger.info(f"Recommended Hardware Type: {recommended_config['type']}")

    install_pytorch_version(recommended_config['torch'])

    # Determine ASR and TTS device values
    asr_device_val = "cuda" if recommended_config['type'] == "cuda" else "cpu"
    tts_device_val = "cpu"  # TTS device is always CPU

    logger.info(f"Determined ASR device: {asr_device_val}")
    logger.info(f"Determined TTS device: {tts_device_val}")

    env_settings = {
        "TORCH_VARIANT": recommended_config['torch'],
        "DEVICE_TYPE": recommended_config['type'],
        "CUDA_DEVICE_NAME": cuda_device_name, # Name from torch.cuda.get_device_name
        "CPU_VENDOR": detect_cpu_vendor()
    }
    if recommended_config['type'] == "cuda" and 'cuda' in recommended_config:
        env_settings["RECOMMENDED_CUDA_VERSION"] = recommended_config['cuda']

    write_env_file(env_settings, base_dir, asr_device=asr_device_val, tts_device=tts_device_val)
    logger.info("Hardware detection and PyTorch setup complete. Configuration saved to .env file.")

if __name__ == "__main__":
    # Setup basic logging for standalone script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # Determine a base directory (e.g., script's parent directory or CWD)
    # For standalone, .env will be in CWD.
    current_working_directory = Path.cwd()
    run_device_setup(str(current_working_directory))
