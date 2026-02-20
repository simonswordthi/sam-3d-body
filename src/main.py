import sys
import logging
from pathlib import Path

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.app import run_app


if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)