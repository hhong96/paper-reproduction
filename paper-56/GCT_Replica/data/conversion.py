import gzip
import shutil

input_file = 'treatment.csv.gz'
output_file = 'treatment.csv'

with gzip.open(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Decompressed {input_file} to {output_file}")
