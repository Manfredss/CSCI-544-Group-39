import arxiv, os, re, requests, tqdm, warnings
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader, PdfWriter

warnings.filterwarnings('ignore')

# -----------------------------
# 通用辅助函数
# -----------------------------
# def normalize_title(title: str):
#     title = title.lower().replace('\n', ' ')
#     title = re.sub(r'\s+', ' ', title).strip()
#     return title

def fetch_arxiv_papers(query: str):
    search = arxiv.Search(
        query=query,
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    for result in search.results():
        return {"title": result.title,
                "authors": [a.name for a in result.authors],
                "pdf_url": result.pdf_url}

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

def download_and_clean_pdf(pdf_url, save_dir="papers"):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, os.path.basename(pdf_url) + ".pdf")

    r = requests.get(pdf_url)
    with open(pdf_path, "wb") as f:
        f.write(r.content)

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    reference_start_page = None
    
    # 查找 Reference 开始的页面
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        
        # 查找 Reference 相关的关键词
        if re.search(r'\b(References|Bibliography|REFERENCE|BIBLIOGRAPHY)\b', text, re.IGNORECASE):
            # 检查该页是否还有其他主要内容
            text_before_ref = re.split(r'References|Bibliography', text, flags=re.IGNORECASE)[0]
            if len(text_before_ref.strip()) > 200:  # 如果该页还有较多其他内容
                reference_start_page = page_num + 1  # 标记下一页开始删除
            else:
                reference_start_page = page_num  # 标记当前页开始删除
            break
    
    # 如果没有找到 Reference，保留所有页面
    if reference_start_page is None:
        reference_start_page = len(reader.pages)
    
    # 添加需要保留的页面
    for page_num in range(reference_start_page):
        writer.add_page(reader.pages[page_num])
    
    # 保存结果
    with open(pdf_path, 'wb') as output_file:
        writer.write(output_file)

    # reader = PdfReader(pdf_path)
    # writer = PdfWriter()

    # appendix_page = None
    # appendix_keywords = {'appendix', 'appendices', 'appendixes', 'supplementary'}

    # for page_num in range(len(reader.pages)):
    #     page = reader.pages[page_num]
    #     text = page.extract_text().lower()
    #     if any(keyword in text for keyword in appendix_keywords):
    #         lines = text.split("\n")
    #         if lines and any(keyword in lines[0].lower() for keyword in appendix_keywords):
    #             appendix_page = page_num

    # if not appendix_page:
    #     for page in reader.pages:
    #         writer.add_page(page)
    # else:
    #     for page_num in range(appendix_page):
    #         writer.add_page(reader.pages[page_num])

    # # cleaned_path = pdf_path.replace(".pdf", "_clean.pdf")
    # with open(pdf_path, "wb") as f:
    #     writer.write(f)


if __name__ == "__main__":
    conferences = {
        "cvpr": fetch_cvpr_titles,
        "iclr": fetch_iclr_titles,
        "neurips": fetch_neurips_titles,
        "icml": fetch_icml_titles,
    }

    year = 2024
    for conf_name, fetch_func in conferences.items():
        print(f"\n=== {conf_name.upper()} ===")
        try:

            accepted_titles = fetch_func(year)
            accepted_titles = list(accepted_titles)
            print(f"Loaded {len(accepted_titles)} accepted papers from {conf_name}")

            # 从 arxiv 下载候选论文
            # 每个会议下载前 3 篇做示例
            for title in tqdm.tqdm(accepted_titles[:3], desc=f"Downloading, cleaning and saving papers from {conf_name}"):
                paper = fetch_arxiv_papers(title)
                download_and_clean_pdf(paper["pdf_url"], save_dir=f"papers_{conf_name}")

        except Exception as e:
            print(f"[ERROR] Failed to fetch {conf_name} papers: {e}")
