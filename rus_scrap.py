import requests
from bs4 import BeautifulSoup
import re

# Wikipedia page URL (Russian)
url = 'https://ru.wikipedia.org/wiki/Список_названий_звёзд'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all tables with class "wikitable"
tables = soup.find_all('table', {'class': 'wikitable'})

russian_star_names = []

# Iterate through tables and extract star names (first column)
for table in tables:
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if cols:
            star_name = cols[0].get_text(strip=True)  # First column contains the name
            russian_star_names.append(star_name)

# Remove references in square brackets and filter out names with Latin characters
ru_star_names_clean = []
for name in russian_star_names:
    name = re.sub(r'\[.+?\]', '', name)  # Remove references
    if not re.search(r'[A-Za-z]', name):  # Keep only names without Latin characters
        ru_star_names_clean.append(name)

# Print or save the cleaned star names
print("--"*80)
print(ru_star_names_clean)

# Define the filename
filename = "russian_star_names.txt"

# Save the list to a text file
with open(filename, "w", encoding="utf-8") as file:
    for name in ru_star_names_clean:
        file.write(name + "\n")

print(f"Saved {len(ru_star_names_clean)} star names to {filename}")
