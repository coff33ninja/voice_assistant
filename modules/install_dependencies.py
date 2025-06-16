import subprocess
import platform
import sys
import shutil
import logging

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
        logger.error(
            "ollama is not installed. Install from https://ollama.com/download"
        )
        sys.exit(1)

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
        logger.info(
            "Ensure Visual C++ Build Tools are installed for Windows: "
            "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
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

    # Step 3: Install compatible torch versions
    logger.info("Installing PyTorch CPU versions...")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.5.0+cpu",
            "torchvision==0.20.0+cpu",
            "torchaudio==2.5.0+cpu",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        "Failed to install PyTorch",
    )

    # Step 4: Install pyaudio (for precise-runner)
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

    # Step 5: Install core dependencies
    logger.info("Installing core Python dependencies...")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "whisperx",
            "TTS==0.22.0",
            "langchain==0.3.1",
            "langchain-community==0.3.1",
            "transformers==4.45.2",
            "precise-runner==0.2.1",
            "pvporcupine==3.0.3",
            "datasets==3.0.0",
            "accelerate==1.0.0",
            "sounddevice",
            "aiohttp",
        ],
        "Failed to install core dependencies",
    )


def pull_ollama_model():
    """Pull llama2 model for ollama."""
    logger.info("Pulling llama2 model for ollama...")
    run_command(
        ["ollama", "pull", "llama2"],
        "Failed to pull llama2 model. Ensure ollama is running: ollama serve",
    )


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
