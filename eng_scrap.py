import requests
from bs4 import BeautifulSoup
import re

# Wikipedia page URL (Russian)
url = 'https://ru.wikipedia.org/wiki/Список_названий_звёзд'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all tables with class "wikitable"
tables = soup.find_all('table', {'class': 'wikitable'})

star_names = []

# Iterate through tables and extract star names (first column)
for table in tables:
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if cols:
            star_name = cols[1].get_text(strip=True)  # First column contains the name
            star_names.append(star_name)

# Print or save the star names
print(star_names)

star_names_clean = []
for name in star_names:
    name = re.sub(r'\[.+?\]', '', name)
    star_names_clean.append(name)

print("--"*80)
print(star_names_clean)

# Define the filename
filename = "eng_star_names.txt"

# Save the list to a text file
with open(filename, "w", encoding="utf-8") as file:
    for name in star_names_clean:  # or use `star_names` for the English list
        file.write(name + "\n")  # Write each name on a new line

print(f"Saved {len(star_names_clean)} star names to {filename}")
