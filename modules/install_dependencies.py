import subprocess
import platform
import sys
import shutil
import logging
import os # Added
from dotenv import set_key # Added

from modules.config import (
    LLM_MODEL_NAME,
    _PROJECT_ROOT,
)  # Import _PROJECT_ROOT for .env path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if required tools and Python version are installed."""
    # Check Python version (3.7–3.11 for TTS compatibility)
    python_version = sys.version_info
    if not (3, 7) <= (python_version.major, python_version.minor) <= (3, 11):
        logger.error(
            f"Python {python_version.major}.{python_version.minor} is not supported. "
            "Use Python 3.7–3.11. Download from https://www.python.org/"
        )
        sys.exit(1)
    logger.info(f"Python {python_version.major}.{python_version.minor} detected.")

    # Check pip
    try:
        subprocess.check_output([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        logger.error("pip is not installed. Run: python -m ensurepip --upgrade")
        sys.exit(1)

    # Check system tools
    system = platform.system()
    if system == "Linux":
        if not shutil.which("apt"):
            logger.error("apt is required on Linux. Use a Debian-based distribution.")
            sys.exit(1)
    elif system == "Darwin":
        if not shutil.which("brew"):
            logger.error(
                'Homebrew is not installed. Run: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            )
            sys.exit(1)
    elif system == "Windows":
        if shutil.which("winget"):
            logger.info("winget detected for FFmpeg installation.")
        else:
            logger.warning("winget not found. FFmpeg may need manual installation.")

    # Check ollama
    if not shutil.which("ollama"):
        logger.warning(
            "ollama executable not found. LLM capabilities will be unavailable. "
            "Install from https://ollama.com/download if you need LLM features."
        )

    # Check if running in a virtual environment
    if (
        not hasattr(sys, "real_prefix")
        and not getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    ):
        logger.warning(
            "Not running in a virtual environment. This may cause dependency conflicts. "
            "Run: python -m venv .venv && .venv\\Scripts\\activate (Windows) or source .venv/bin/activate (Linux/macOS)"
        )


def run_command(cmd, error_msg="Command failed"):
    """Run a subprocess command with error handling."""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"{error_msg}: {e.stderr}")
        raise


def install_system_dependencies():
    """Install system dependencies (ffmpeg, libsndfile, portaudio)."""
    system = platform.system()
    logger.info(f"Installing system dependencies for {system}...")

    if system == "Linux":
        run_command(["sudo", "apt", "update"], "Failed to update apt package list")
        run_command(
            [
                "sudo",
                "apt",
                "install",
                "-y",
                "ffmpeg",
                "libsndfile1",
                "portaudio19-dev",
            ],
            "Failed to install system dependencies on Linux",
        )
    elif system == "Darwin":
        run_command(
            ["brew", "install", "ffmpeg", "libsndfile", "portaudio"],
            "Failed to install system dependencies on macOS",
        )
    elif system == "Windows":
        # Check for ffmpeg
        if shutil.which("ffmpeg"):
            logger.info("FFmpeg is already installed.")
        else:
            try:
                run_command(
                    ["winget", "install", "ffmpeg"], "winget failed to install FFmpeg"
                )
            except subprocess.CalledProcessError:
                logger.warning(
                    "Please download FFmpeg from https://ffmpeg.org/download.html, "
                    "extract it, and add the 'bin' directory to your PATH."
                )

        # Recommend Visual C++ Build Tools and Chocolatey for libsndfile/portaudio
        logger.warning(
            "For runtime dependencies like ONNXRuntime (used by WhisperX), ensure the Microsoft Visual C++ Redistributable is installed.\n"
            "Download from the official Microsoft site (Visual Studio 2015-2022 section):\n"
            "  - For 64-bit systems: https://aka.ms/vs/17/release/vc_redist.x64.exe"
            "  - For 32-bit systems: https://aka.ms/vs/17/release/vc_redist.x86.exe"
            "You can typically install the x64 version on modern systems. If unsure, you can install both."
        )
        if shutil.which("choco"):
            logger.info("Installing libsndfile, portaudio, and espeak-ng via Chocolatey...")
            try:
                run_command(
                    ["choco", "install", "libsndfile", "portaudio", "espeak-ng", "-y"],
                    "Failed to install libsndfile/portaudio/espeak-ng via Chocolatey",
                )
            except subprocess.CalledProcessError:
                logger.warning(
                    "Chocolatey installation failed. Install libsndfile, portaudio, and espeak-ng manually or ensure they are in PATH."
                )
        else:
            logger.warning(
                "Chocolatey not found. Install libsndfile and portaudio manually or use: "
                "Set-ExecutionPolicy Bypass -Scope Process -Force; "
                "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
                "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
            )


def install_python_dependencies():
    """Install Python dependencies."""
    logger.info("=== Starting Python Dependency Installation ===")

    # Step 1: Upgrade pip and wheel
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        "Failed to upgrade pip and wheel",
    )

    # Step 2: Clean conflicting packages
    logger.info("Uninstalling conflicting packages...")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio",
            "whisperx",
        ],
        "Failed to uninstall conflicting packages",
    )

    # Step 3: Install pyaudio (for precise-runner)
    # Note: PyTorch installation is now handled by the 'device_detection' step in setup_assistant.py
    logger.info("Installing pyaudio...")
    try:
        run_command(
            [sys.executable, "-m", "pip", "install", "pyaudio==0.2.14"],
            "Failed to install pyaudio",
        )
    except subprocess.CalledProcessError:
        if platform.system() == "Windows":
            logger.info("Attempting to install pyaudio via pipwin...")
            run_command(
                [sys.executable, "-m", "pip", "install", "pipwin"],
                "Failed to install pipwin",
            )
            run_command(
                [sys.executable, "-m", "pipwin", "install", "pyaudio"],
                "Failed to install pyaudio via pipwin. Install portaudio manually: choco install portaudio",
            )

    # Step 4: Install core dependencies
    logger.info("Installing core Python dependencies...")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "whisperx",
            "onnxruntime==1.17.3",  # Pin onnxruntime version
            "TTS==0.22.0",
            "langchain==0.3.1",
            "langchain-community==0.3.1",
            "transformers==4.45.2",
            # "precise-runner==0.2.1", # Temporarily commented out due to Python 3.11 incompatibility
            "pvporcupine==3.0.3",
            "datasets==3.0.0",
            "accelerate==1.0.0",
            "sounddevice",
            "aiohttp",
            "python-dotenv",
            "nest_asyncio",
            "ics",
            "dateparser",
            "pyspellchecker",  # Added spell-checking library
            "watchdog",  # Added for file watching
            "onnx",
            "pytest",
            "pytest-asyncio",
            "pytest-twisted",
            "pytest-trio",
            "pytest-cov",
            "pytest-tornasync",
            "twisted",
        ],
        "Failed to install core dependencies",
    )

    # Step 4b: Install OpenAI Whisper (standard)
    logger.info("Installing OpenAI Whisper (standard)...")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "openai-whisper-20231117"
        ],
        "Failed to install OpenAI Whisper (standard)",
    )

def list_ollama_models() -> list[str]:
    """Lists locally available Ollama models."""
    logger.info("Checking for locally available Ollama models...")
    try:
        # Pass an empty dict for env if you don't want to inherit the current environment
        # or if specific env vars might interfere. Usually, inheriting is fine.
        result = run_command(["ollama", "list"], "Failed to list Ollama models. Is Ollama server running?")
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1: # Only header or empty
            logger.info("No Ollama models listed by 'ollama list'.")
            return []
        models = []
        for line in lines[1:]: # Skip header
            parts = line.split() # Splits by whitespace
            if parts:
                models.append(parts[0]) # First part is NAME (e.g., llama2:latest)
        return models
    except Exception as e:
        logger.warning(f"Could not retrieve Ollama models list: {e}. Assuming no models are available locally.")
        return []

def pull_ollama_model():
    """Interactively select and pull an Ollama LLM model, updating .env."""
    if not shutil.which("ollama"):
        logger.warning("Ollama executable not found. Skipping Ollama LLM model setup.")
        # Ensure LLM_MODEL_NAME is set to a non-functional value or default if not already present
        return # Exit function if ollama is not available

    # Ensure _PROJECT_ROOT is an absolute and canonical path
    resolved_project_root = os.path.realpath(_PROJECT_ROOT)
    if not os.path.isabs(resolved_project_root): # Double check after realpath, though realpath usually makes it absolute
        raise ValueError(f"Invalid project root path after realpath: {resolved_project_root}. It must be an absolute path.")
    sanitized_project_root = resolved_project_root # Use the resolved path
    dotenv_path = os.path.join(sanitized_project_root, ".env")
    if not os.path.exists(dotenv_path):
        open(dotenv_path, 'a').close() # Ensure .env file exists for set_key

    initial_effective_model = LLM_MODEL_NAME # From config.py (respects .env or defaults)
    logger.info("--- Ollama LLM Model Configuration ---")
    logger.info(f"The current effective LLM model (from .env or default) is: {initial_effective_model}")

    available_models = list_ollama_models()
    if available_models:
        logger.info("Locally available Ollama models:")
        for model in available_models:
            logger.info(f"  - {model}")
    else:
        logger.info("No Ollama models found locally (or 'ollama list' failed).")

    chosen_model_for_session = initial_effective_model

    while True: # Loop for model selection and pulling
        prompt_message = (
            f"\nEnter the Ollama model name you want to use (e.g., 'llama2', 'mistral:latest').\n"
            f"- If listed as available, it will be selected.\n"
            f"- If not listed, an attempt will be made to pull it.\n"
            f"- Press Enter to try using/pulling the current effective model: '{initial_effective_model}': "
        )
        user_input = input(prompt_message).strip()
        selected_model_name = user_input if user_input else initial_effective_model

        if selected_model_name in available_models:
            logger.info(f"Selected locally available model: '{selected_model_name}'")
            chosen_model_for_session = selected_model_name
            break
        else:
            logger.info(f"Model '{selected_model_name}' is not available locally. Attempting to pull...")
            try:
                run_command(
                    ["ollama", "pull", selected_model_name],
                    f"Failed to pull '{selected_model_name}'. Ensure Ollama is running ('ollama serve') and the model name is valid.",
                )
                logger.info(f"Successfully pulled '{selected_model_name}'. This model will be used.")
                chosen_model_for_session = selected_model_name
                break
            except Exception: # run_command raises on failure, error message already logged by it
                retry_choice = input(f"Pulling '{selected_model_name}' failed. (r)etry with the same or different name, or (s)kip LLM model setup for now? [r/s]: ").strip().lower()
                if retry_choice == 's':
                    logger.warning("Skipping Ollama LLM model setup. The assistant might not have LLM capabilities if the initial model was not available or couldn't be pulled.")
                    return # Exit function, .env won't be updated by this interaction

    env_value_before_set = os.getenv("LLM_MODEL_NAME")
    if chosen_model_for_session != initial_effective_model or env_value_before_set != chosen_model_for_session:
        if set_key(dotenv_path, "LLM_MODEL_NAME", chosen_model_for_session, quote_mode="always"):
            logger.info(f"LLM model name set to '{chosen_model_for_session}' in {dotenv_path}")
            os.environ["LLM_MODEL_NAME"] = chosen_model_for_session # Update current session's env
        else:
            logger.warning(f"Could not save LLM_MODEL_NAME to {dotenv_path}. Using '{chosen_model_for_session}' for this session only.")
    else:
        logger.info(f"LLM model '{chosen_model_for_session}' is already configured or matches the default. No .env update needed for LLM_MODEL_NAME.")


def test_tts_installation():
    """Test TTS installation."""
    logger.info("Testing TTS installation...")
    try:
        from TTS.api import TTS  # type: ignore

        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        tts.tts_to_file(text="Test successful.", file_path="test_tts.wav")
        logger.info("TTS test successful. Output saved to test.wav")
    except Exception as e:
        logger.error(f"TTS test failed: {str(e)}")
        logger.warning(
            "TTS installation may have issues. Check dependencies and system libraries."
        )


def install_dependencies():
    """Install all dependencies for the voice assistant project."""
    try:
        check_prerequisites()
        install_system_dependencies()
        install_python_dependencies()
        pull_ollama_model()
        test_tts_installation()
        logger.info("=== Dependency Installation Completed Successfully ===")
    except Exception as e:
        logger.error(f"Installation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    install_dependencies()
