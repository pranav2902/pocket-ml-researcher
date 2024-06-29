import requests
from bs4 import BeautifulSoup as bs
from arxiv import Client, Search
import os
from tqdm import tqdm
import logging
import sys

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_links(webpage: str) -> list[str]:
    r = requests.get(webpage)
    soup = bs(r.text, 'lxml')
    selector = soup.select('ul > li > a')
    papers = []
    logger.info("Getting links")
    for link in selector:
        href = link.get('href')
        if (str(href[0]) not in ["/", "#"]) and ("github" not in str(href)):
            papers.append(href)
    return papers


def get_pdfs(links, subdir):
    """
    Downloads articles/pdfs from given links to subdir
    :param links: List of strings containing links to pdfs
    :param subdir: Location to store pdfs
    :return:
    """
    # Get ids for Arxiv links
    arxiv_ids = []
    logger.info("Getting IDs from links")
    for link in links:
        if "arxiv" in link:
            id = link.split("/")[-1]
            arxiv_ids.append(str(id))

    # Create subdir if not exists
    if subdir not in os.listdir("../"):
        os.makedirs(subdir)

    client = Client()
    logger.info("Downloading PDFs")
    for id in tqdm(arxiv_ids):
        if not os.path.isfile("../" + subdir + f"/arxiv-{id}.pdf"):
            paper = next(client.results(Search(id_list=[id])))
            paper.download_pdf(dirpath="../" + subdir, filename=f"arxiv-{id}.pdf")


if __name__ == "__main__":
    webpage = "https://github.com/aimerou/awesome-ai-papers?tab=readme-ov-file"
    paper_links = get_links(webpage)
    get_pdfs(paper_links,"arxiv_papers")
    sys.exit()