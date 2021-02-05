import pathlib
import webbrowser
import random
import time
import threading
from stb.visualizer_app.main import main as visualizer_app_main


def open_browser_later(port):
    time.sleep(5)
    webbrowser.open(f"http://localhost:{port}")


def main():
    port = str(random.randint(64000, 65000))
    data_dir = str(pathlib.Path(__file__).parent / "data")
    threading.Thread(target=open_browser_later, args=[port]).start()
    visualizer_app_main(["--storage-dir", data_dir, "--port", port])


if __name__ == "__main__":
    main()
