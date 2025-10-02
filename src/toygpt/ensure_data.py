import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data")
DEST = os.path.join(DATA_DIR, "input.txt")

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DEST):
        print(f"Dataset already exists at {DEST}")
        return
    print(f"Downloading Tiny Shakespeare to {DEST} ...")
    urllib.request.urlretrieve(URL, DEST)
    print("Done.")

if __name__ == "__main__":
    main()