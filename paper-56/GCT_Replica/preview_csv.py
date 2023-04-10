import pandas as pd
import sys

def preview_csv(file_path, n_rows=10):
    data = pd.read_csv(file_path, nrows=n_rows)
    print(data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_csv.py <file_path> [<n_rows>]")
        sys.exit(1)

    file_path = sys.argv[1]
    n_rows = 10
    if len(sys.argv) > 2:
        n_rows = int(sys.argv[2])

    preview_csv(file_path, n_rows)
