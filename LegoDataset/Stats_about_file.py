import pandas as pd
import re
import os
# Load the CSV file
file_path = 'lego_sets_and_themes.csv'  # Update this with the correct file path
data = pd.read_csv(file_path)

# Queries

# 1. Count entries in specific year ranges
year_col = data['year_released']
counts = {
    "x < 2000": len(year_col[year_col < 2000]),
    "2000 < x < 2010": len(year_col[(year_col > 2000) & (year_col < 2010)]),
    "2010 < x < 2020": len(year_col[(year_col > 2010) & (year_col < 2020)]),
    "x > 2020": len(year_col[year_col > 2020]),
    "2020 < x <= 2021": len(year_col[(year_col > 2020) & (year_col <= 2021)]),
    "2021 < x <= 2022": len(year_col[(year_col > 2021) & (year_col <= 2022)])
}
print("Year Ranges:", counts)
print("\n -----------------------------------------------")

# 2. Filter entries based on specific names and count post-2015 and post-2020
# Open the file and read its contents into a list

objects_to_test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'objectstotest.txt')
with open(objects_to_test_file, 'r') as file:
    objectstotest = []
    for line in file.readlines():
        # Split lines containing underscores into separate words and normalize
        words = line.strip().replace("_", " ").split()
        objectstotest.extend(word.lower() for word in words)

matching_images = data[
    data['set_name'].str.lower().str.contains('|'.join(objectstotest), na=False)
]
post_2015_images = matching_images[matching_images['year_released'] > 2015]
post_2020_images = matching_images[matching_images['year_released'] > 2020]
print(" Number of Images for the objects in the objectstotest.txt file")
print("Post-2015 Images:", len(post_2015_images))
print("Post-2020 Images:", len(post_2020_images))
print("\n-----------------------------------------------")
# 3. Filter entries for specific themes and year ranges
interested_themes = ['Archtiecture', 'Skylines', 'Pick A Model', 'Creator 3-in-1']
filtered_entries = data[
    (data['theme_name'].isin(interested_themes)) &
    (data['year_released'].isin([2020, 2021, 2022, 2023]))
]
entry_counts_per_year = filtered_entries['year_released'].value_counts().sort_index()
print(f"Entries for themes {', '.join(interested_themes)} (2020-2023):")
print(entry_counts_per_year)
print("\n-----------------------------------------------")
# 4. Count entries for themes without year filtering


entry_counts_per_theme = data[data['theme_name'].isin(interested_themes)]['theme_name'].value_counts()
print("Entries for Themes Without Year Filtering:")
print(entry_counts_per_theme)


