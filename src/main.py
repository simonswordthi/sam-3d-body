import sys
from ui.app import SAM3DApp

def main():
    try:
        app = SAM3DApp()  # Initialize the SAM3D application
        app.run()  # Run the application
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()