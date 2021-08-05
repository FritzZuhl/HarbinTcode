from bs4 import BeautifulSoup
import re

def strip_html_tags(doc):
    soup = BeautifulSoup(doc, 'html.parser')
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+',  '\n', stripped_text)
    return stripped_text


