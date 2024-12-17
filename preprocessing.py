import pandas as pd
import os

def parse_spectrum(spectrum_string):
    list_of_strings = str(spectrum_string).strip('[]').split(';')
    list_of_floats = [float(s) for s in list_of_strings]
    spectrum = pd.Series(data = list_of_floats, index = [f'Channel_{i}' for i in range(len(list_of_floats))])
    return spectrum

input_folder = "raw_data"
if not os.path.exists(input_folder):
    raise Exception(f"'Folder {input_folder}' does not exist")

csv_pathnames = [input_folder + "/" + name for name in os.listdir(input_folder) if name.endswith('.csv')]
if csv_pathnames == []:
    raise Exception(f"Put .csv files in {input_folder}, it is empty now.")
print("Reading .csv files: ", csv_pathnames)

rows = []
for pathname in csv_pathnames:
    df_with_raw_spectrum = pd.read_csv(pathname)
    for _, row in df_with_raw_spectrum.iterrows():
        spectrum = parse_spectrum(row['Spectrum'])
        other_data = row.drop('Spectrum')
        full_row = pd.concat([other_data, spectrum])
        rows.append(full_row) 

df_wide = pd.DataFrame(rows)
print(df_wide)
output_pathname = 'processed_data/df_full.csv'
df_wide.to_csv(output_pathname, index=False)
print(f"DataFrame saved to {output_pathname}")
