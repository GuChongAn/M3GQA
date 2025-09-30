import os
import json
import requests
import wikipediaapi
from tqdm import tqdm
from bs4 import BeautifulSoup
WMQA_PATH = "../dataset/2WikiMultihopQA"

# input: wikipedia title, wikipedia url
# output: 1) image url and table
#         2) store image and table in [dataset]/images | [dataset]/table.json
def get_image_table(title, url):
    # get wikipedia html
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, verify=False, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    # capture table
    table_json = {"title": title, "tables": []}
    for table in tables:
        rows = table.find_all("tr")
        table_content = []
        for row in rows:
            cols = row.find_all(["th", "td"])
            table_content.append([col.text.strip() for col in cols])
        table_json['tables'].append(table_content)
    with open(os.path.join(WMQA_PATH, "table.jsonl"), 'a+') as f:
        if table_json['tables']:
            f.write("{}\n".format(json.dumps(table_json)))

    # capture image
    images = soup.select('.mw-file-description .mw-file-element')
    image_urls = [] 
    for img in images:
        src = img.get("src")
        height = img.get("height")
        width = img.get("width")
        if (not src or not height or not width) or (int(height) < 60 or int(width) < 60) or ("svg" in src):
            continue
        image_urls.append("https:" + src)
    for i, img_url in enumerate(image_urls):
        img_data = requests.get(img_url, headers=headers)
        with open(os.path.join(WMQA_PATH, f"images/{title}_{i+1}.jpg"), "wb") as f:
            f.write(img_data.content)
    
    return table_json, image_urls

if __name__ == "__main__":
    wiki_wiki = wikipediaapi.Wikipedia('qadataset(hushuhao24@mails.ucas.ac.cn)', 'en')
    with open(os.path.join(WMQA_PATH, "dev.json"), 'r+') as f:
        wikiqa = json.load(f)

    if not os.path.exists(os.path.join(WMQA_PATH, "images")):
        os.makedirs(os.path.join(WMQA_PATH, "images"))

    exist_url = []
    for data in tqdm(wikiqa):
        evidence_titles = [t[0] for t in data['supporting_facts']]
        for title in evidence_titles:
            page = wiki_wiki.page(title)
            if (not page.exists()) or (page.fullurl in exist_url):
                continue
            tables, image_urls = get_image_table(page.title, page.fullurl)
            exist_url.append(page.fullurl)