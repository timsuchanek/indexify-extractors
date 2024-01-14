import os

from typing import List

from urllib.request import urlopen

from langchain.docstore.document import Document


def save_html_pages(urls, path):
    if not os.path.exists(path):
        os.mkdir(path)

    for url in urls:
        response = urlopen(url)

        with open(f"{path}/{url.split('/')[-1]}.html", "wb") as f:
            f.write(response.read())


def parse_html_files(path: str) -> List[Document]:

    html_content = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if file_path.endswith(".html"):
            with open(file_path, "r") as f:
                document = Document(page_content=f.read(),
                                    metadata={"url": filename})
                html_content.append(document)

    return html_content
