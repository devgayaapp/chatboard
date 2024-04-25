from bs4 import BeautifulSoup
import tiktoken
import requests
import random
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse


def extract_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain.split('.')[0]

# https://deviceatlas.com/blog/list-of-user-agent-strings
def get_random_user_agent():
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
        # "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
    ]
    return random.choice(USER_AGENTS)
    

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens




def get_head_meta(soup):
    meta_text = ""
    for head in soup.find_all('head'):
        for tag in head.find_all('meta'):
            if tag.find_all('script'):
                continue
            if tag.find_all('style'):
                continue
            if tag.find_all('link'):
                continue
            meta_text += str(tag) + "\n"
    return meta_text


# def split_html_to_chunk(html_text):
#     soup = BeautifulSoup(html_text, 'html.parser')
#     for scrypt_or_style in soup(['script', 'style']):
#         scrypt_or_style.decompose()
#     meta_text = get_head_meta(soup)

TAG_LIST = [
        "aria-label",
        "aria-labelledby",
        "aria-hidden",
    ]

class HtmlTag:

    def __init__(self, tag):
        self.tag = tag

    def type(self):
        return self.tag.name
    
    def is_header(self):
        return self.tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]
    
    def get_text(self, remove_tags=True):
        tag = self.tag
        att = [f"<{a}={tag.get(a)}" for a in TAG_LIST if tag.get(a)]    
        start_tag = f"<{tag.name} {' '.join(att)}>" if att else f"<{tag.name}>"            
        tag_content =  "\n".join(phrase.strip() for phrase in tag.get_text().strip().split("  "))  

        if not remove_tags:
            tag_str = start_tag + " " + tag_content+ " " + f"</{tag.name}>"
        else:
            tag_str = tag_content
        return tag_str


class HtmlChunk:

    def __init__(self, tags=None):
        self.tags = tags if tags else []
        self.stats = None

    def append(self, tag):
        self.tags.append(tag)

    @property
    def text(self):
        return self.get_text()
    
    def get_text(self, remove_tags=True):
        return "\n".join([tag.get_text(remove_tags) for tag in self.tags])
    
    @property
    def num_tokens(self):
        self._clac_stats()        
        return self.stats['num_tokens']
    
    @property
    def num_unique_tokens(self):
        self._clac_stats()
        return self.stats['num_unique_tokens']
    
    @property
    def ttr(self):
        self._clac_stats()
        return self.stats['ttr']
    
    
    def _clac_stats(self):
        if self.stats is not None:
            return
        text = self.get_text()
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()] 
        unique_words = set(tokens)
        num_unique_words = len(unique_words)
        total_tokens = len(tokens)
        ttr = num_unique_words / (total_tokens + 1)
        self.stats = {
            "num_tokens": total_tokens,
            "num_unique_tokens": num_unique_words,
            "ttr": ttr
        }
    
    
    


    def __getitem__(self, index):
        return self.tags[index]
    
    def __len__(self):
        return len(self.tags)
    
    def __iter__(self):
        return iter(self.tags)
    
    def __str__(self):
        return "\n".join([tag.get_text() for tag in self.tags])
    
    def html_reper(self):
        return "\n".join([str(tag) for tag in self.tags])


class HtmlDoc:

    def __init__(self, html_text) -> None:
        soup = BeautifulSoup(html_text, 'html.parser')
        self.soup = soup
        for scrypt_or_style in soup(['script', 'style']):
            scrypt_or_style.decompose()
        self.chunk_list = []
        self.meta_text = get_head_meta(soup)        
        tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'img', 'title'])        
        curr_chunk = HtmlChunk()
        for tag in tags:
            soup_tag = HtmlTag(tag)
            if soup_tag.is_header():
                self.append(curr_chunk)
                curr_chunk = HtmlChunk()
            curr_chunk.append(soup_tag)
        if self.chunk_list[-1] != curr_chunk:
            self.chunk_list.append(curr_chunk)

    def append(self, chunk):
        self.chunk_list.append(chunk)

    def get_text(self, remove_tags=True, filter_content=True):
        if filter_content:
            return "\n".join([chunk.get_text(remove_tags) for chunk in self.iter_content()])
        return "\n".join([chunk.get_text(remove_tags) for chunk in self.chunk_list])
    
    @property
    def text(self):
        return self.get_text()
    
    def get_html(self):
        return "\n".join([str(tag) for tag in self.chunk_list])
    
    def get_meta(self):
        return self.meta_text
    
    def __getitem__(self, index):
        return self.chunk_list[index]
    
    def __len__(self):
        return len(self.chunk_list)
    
    def iter_content(self):
        return iter([c for c in self.chunk_list if c.num_tokens > 30 and c.num_unique_tokens > 20])
    
    def __iter__(self):
        return iter(self.chunk_list)
    
    def __str__(self):
        return self.get_text()
    
    def to_pandas(self):
        import pandas as pd
        data = []
        for chunk in self.chunk_list:
            chunk_data ={
                "text": chunk.get_text(),
                "html": chunk.html_reper(),
                # "stats": chunk.stats()
            }
            chunk_data.update(chunk.stats())
            data.append(chunk_data)
        return pd.DataFrame(data)
    






def get_url_content(url, is_process=True, remove_tags=True):
    response = requests.get(
        url,
        headers={
            "User-Agent": get_random_user_agent()
        }
    )
    if response.status_code == 200:
        page_content = response.content
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")    
        raise Exception(f"Failed to retrieve the page. Status code: {response.status_code}")
    
    if is_process:
        return HtmlDoc(page_content)
        # return process_html_page(response.content, remove_tags=remove_tags)
    else:
        return response.content
    # soup = BeautifulSoup(page_content, 'html.parser')
    # article_content = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    # return "\n".join([p.get_text() for p in article_content])
