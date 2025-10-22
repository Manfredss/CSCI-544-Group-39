import re, requests
from bs4 import BeautifulSoup

def fetch_cvpr_titles(year: int):
    url = f"https://openaccess.thecvf.com/CVPR{year}?day=all"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.Session().get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=lambda x: x and 'pdf' in x)
        titles = set()
        for link in links:
            link = link.get('href')
            paper_name = re.search(r'/([^/]+)\.pdf$', link)
            title = paper_name.group(1)
            if 'supplement' in title: 
                continue
            title = title.split('_CVPR_')[0]
            title = title.replace('_', ' ').split()
            title = ' '.join(title[1:])
            titles.add(title)
        return titles
    except Exception as e:
        print(f"爬取 CVPR 过程中出现错误: {e}")
        return []

def fetch_iclr_titles(year: int):
    url = f"https://iclr.cc/virtual/{year}/papers.html?filter=titles"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.Session().get(url, headers=headers)
        response.raise_for_status()
        response = response.content.decode('utf-8')
        soup = BeautifulSoup(response, 'html.parser')
        links = soup.find_all('a', href=lambda x: x and f'/virtual/{year}/poster' in x)
        titles = set()
        for link in links:
            title = link.get_text()
            titles.add(title)
        return titles
    except Exception as e:
        print(f"爬取 CVPR 过程中出现错误: {e}")
        return []

def fetch_neurips_titles(year: int):
    url = f"https://neurips.cc/virtual/{year}/papers.html?filter=titles"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.Session().get(url, headers=headers)
        response.raise_for_status()
        response = response.content.decode('utf-8')
        soup = BeautifulSoup(response, 'html.parser')
        links = soup.find_all('a', href=lambda x: x and f'/virtual/{year}/poster' in x)
        titles = set()
        for link in links:
            title = link.get_text()
            titles.add(title)
        return titles
    except Exception as e:
        print(f"爬取 CVPR 过程中出现错误: {e}")
        return []

def fetch_icml_titles(year: int):
    url = f"https://icml.cc/virtual/{year}/papers.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.Session().get(url, headers=headers)
        response.raise_for_status()
        response = response.content.decode('utf-8')
        soup = BeautifulSoup(response, 'html.parser')
        links = soup.find_all('a', href=lambda x: x and f'/virtual/{year}/poster' in x)
        titles = set()
        for link in links:
            title = link.get_text()
            titles.add(title)
        return titles
    except Exception as e:
        print(f"爬取 CVPR 过程中出现错误: {e}")
        return []