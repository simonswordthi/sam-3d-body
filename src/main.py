import os
import sys

# Ensure the src/ directory is on the path so relative imports work whether
# the script is launched from the repo root or from inside src/.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ui.app import SAM3DApp


def main() -> None:
    try:
        app = SAM3DApp()
        app.run()
    except Exception as exc:
        print(f"An error occurred: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
