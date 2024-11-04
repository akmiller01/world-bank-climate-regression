# Required libraries
import time
import json
import pandas as pd
import requests
from tqdm import tqdm


# Sectors to select
select_sectors = ["Climate change", "Adaptation", "Mitigation"]

# Function to parse projects
def parse_projects(projects, select_sectors):
    proj_list = []

    for project_id in projects.keys():
        project = projects[project_id]
        themes = project.pop('theme_list', [])
        theme2s = [item for theme in themes if 'theme2' in theme for item in theme['theme2']]
        theme3s = [item for theme in theme2s if 'theme3' in theme for item in theme['theme3']]

        # Remove nested themes from parents
        theme_df = pd.DataFrame(themes)
        theme2_df = pd.DataFrame(theme2s)
        theme3_df = pd.DataFrame(theme3s)

        theme_df = theme_df.drop(columns='theme2', errors='ignore')
        theme2_df = theme2_df.drop(columns='theme3', errors='ignore')
        
        all_themes = pd.concat([theme_df, theme2_df, theme3_df], ignore_index=True).fillna(0)

        # Skip if the project has no sector information at all
        if 'percent' not in all_themes.columns:
            continue
        if all_themes['percent'].astype('float').sum() == 0:
            continue

        if 'name' in all_themes.columns:
            all_themes = all_themes[all_themes['name'].isin(select_sectors)]

        proj_df = pd.DataFrame([project])

        if not all_themes.empty:
            for _, theme in all_themes.iterrows():
                theme_name = theme.get('name', None)
                theme_pct = theme.get('percent', 0)
                if theme_name:
                    proj_df[theme_name] = theme_pct

        proj_list.append(proj_df)

    all_proj_dat = pd.concat(proj_list, ignore_index=True) if proj_list else pd.DataFrame()
    return all_proj_dat

# Initial setup for data retrieval
rows = 500
base_url = "https://search.worldbank.org/api/v3/projects?format=json&fl=id,fiscalyear,project_name,project_abstract,pdo,theme_list&apilang=en&rows="
expected_length = int(json.loads(requests.get(base_url + "0").text)['total'])
expected_pages = -(-expected_length // rows)  # Ceiling division

# Data collection
data_list = []
offset = 0
for i in tqdm(range(expected_pages), desc="Fetching data"):
    page_url = f"{base_url}{rows}&os={offset}"
    results = json.loads(requests.get(page_url).text)
    projects = results.get('projects', [])
    all_proj_dat = parse_projects(projects, select_sectors)
    if not all_proj_dat.empty:
        data_list.append(all_proj_dat)
    offset += rows
    time.sleep(5)

# Combine all project data
wb_climate = pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()

# Filter columns
wb_climate = wb_climate[["id", "proj_id", "fiscalyear", "project_name", "pdo", "project_abstract"] + select_sectors]

# Replace NA with 0 and convert percentages
for sector in select_sectors:
    wb_climate[sector] = wb_climate[sector].fillna(0).astype(float) / 100

# Write to CSV
wb_climate.to_csv("input/wb_api_climate_percentages.csv", index=False)
