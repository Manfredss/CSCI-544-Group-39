import arxiv
import os, re, tqdm, traceback, warnings
import shutil, tarfile, tempfile
from utils import *

warnings.filterwarnings('ignore')


class ArxivLatexExtractor:
    def __init__(self):
        self.client = arxiv.Client()
        
    def search_by_title(self, title):
        """根据标题搜索arXiv文章"""
        search = arxiv.Search(query=title,
                              max_results=1,
                              sort_by=arxiv.SortCriterion.Relevance)
        
        results = list(self.client.results(search))
        return results
    
    def download_latex_source(self, paper, output_dir=None):
        """下载arXiv文章的LaTeX源代码"""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix=f"arxiv_{paper.get_short_id()}_")
        
        # 下载源文件
        paper.download_source(dirpath=output_dir, filename='source.tar.gz')
        tar_path = os.path.join(output_dir, 'source.tar.gz')
        return tar_path, output_dir
    
    def extract_tar_file(self, tar_path, output_dir):
        """解压tar.gz文件"""
        extracted_files = []
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(output_dir)
                extracted_files = tar.getnames()
        except Exception as e1:
            # 如果不是gz格式，尝试普通tar格式
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(output_dir)
                    extracted_files = tar.getnames()
            except Exception as e2:
                print(f"解压失败: {e2}")
                return []
        
        return extracted_files
    
    def find_latex_and_images(self, directory):
        """在目录中查找所有LaTeX文件"""
        latex_files = []
        image_files = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.eps', '.pdf']
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(('.tex', '.ltx', '.latex')):
                    latex_files.append(file_path)
                elif any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file_path)
        
        return latex_files, image_files
    
    def extract_image_references(self, latex_file_path):
        """从LaTeX文件中提取图片引用"""
        try:
            with open(latex_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f'读取文件 {latex_file_path} 失败: {e}')
            return []
        
        # 匹配各种LaTeX图片命令
        patterns = [
            r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}',
            r'\\begin\{figure\}(?:.*?)\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}.*?\\end\{figure\}',
            r'\\epsfig\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}',
            r'\\psfig\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}',
        ]
        
        image_refs = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                # 清理文件名（移除路径和扩展名）
                filename = os.path.splitext(os.path.basename(match))[0]
                image_refs.append(filename)
        
        return list(set(image_refs))  # 去重
    
    def match_images_with_references(self, image_refs, image_files):
        """匹配图片引用和实际图片文件"""
        matched_images = []
        
        for image_file in image_files:
            filename = os.path.splitext(os.path.basename(image_file))[0]
            
            # 直接匹配
            if filename in image_refs:
                matched_images.append(image_file)
                continue
            
            # 模糊匹配：去掉数字后缀
            base_name = re.sub(r'\d+$', '', filename)
            for ref in image_refs:
                ref_base = re.sub(r'\d+$', '', ref)
                if base_name and base_name == ref_base:
                    matched_images.append(image_file)
                    break
        
        return matched_images
    
    def process_article_by_title(self, title, article_output_dir):
        """主处理函数：根据标题处理文章"""
        print(f"搜索文章: {title}")

        results = self.search_by_title(title)
        
        # 搜索文章
        results = self.search_by_title(title)
        
        if not results:
            print("未找到相关文章")
            return None
        
        print(f"找到 {len(results)} 个结果:")
        for i, paper in enumerate(results):
            print(f"{i+1}. {paper.title} (ID: {paper.get_short_id()})")
        
        # 选择第一个结果
        paper = results[0]
        arxiv_id = paper.get_short_id()
        print(f"\n处理文章: {paper.title}")
        print(f"arXiv ID: {paper.get_short_id()}")
        
        # 创建输出目录
        output_dir = article_output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 下载LaTeX源代码
        print("下载LaTeX源代码...")
        tar_path, source_dir = self.download_latex_source(paper, output_dir)
        
        # 解压源代码
        print('解压源代码...')
        extracted_files = self.extract_tar_file(tar_path, source_dir)
        print(f"解压了 {len(extracted_files)} 个文件")

        # 查找文件
        latex_files, all_image_files = self.find_latex_and_images(source_dir)
        print(f"找到 {len(latex_files)} 个LaTeX文件")
        print(f"找到 {len(all_image_files)} 个图片文件")
        
        # 从主LaTeX文件中提取图片引用
        all_image_refs = []
        for latex_file in latex_files:
            refs = self.extract_image_references(latex_file)
            all_image_refs.extend(refs)
            if refs:
                print(f"从 {os.path.basename(latex_file)} 中提取到 {len(refs)} 个图片引用")
        
        print(f"提取到 {len(all_image_refs)} 个图片引用")
        
        # 匹配图片
        matched_images = self.match_images_with_references(all_image_refs, all_image_files)
        print(f"匹配到 {len(matched_images)} 个图片文件")
        
        # 创建图片输出目录
        images_dir = os.path.join(output_dir, "extracted_images")
        os.makedirs(images_dir, exist_ok=True)
        
        # 复制匹配的图片到输出目录
        for img_path in matched_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(images_dir, img_name)
            
            # 如果目标文件已存在，添加后缀
            counter = 1
            base_name, ext = os.path.splitext(img_name)
            while os.path.exists(dest_path):
                dest_path = os.path.join(images_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            shutil.copy2(img_path, dest_path)
        
        # 保存处理结果信息
        self.save_processing_info(paper, output_dir, latex_files, all_image_files, matched_images, all_image_refs, images_dir)
        
        return {
            'paper': paper,
            'arxiv_id': arxiv_id,
            'source_dir': source_dir,
            'latex_files': latex_files,
            'all_images': all_image_files,
            'matched_images': matched_images,
            'image_references': all_image_refs,
            'images_output_dir': images_dir
        }

    def save_processing_info(self,
                             paper, 
                             output_dir, 
                             latex_files, 
                             all_images,
                             matched_images,
                             image_refs,
                             images_dir):
        """保存处理信息到文件"""
        info_file = os.path.join(output_dir, "processing_info.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"标题: {paper.title}\n")
            f.write(f"arXiv ID: {paper.get_short_id()}\n")
            f.write(f"发布时间: {paper.published}\n")
            f.write(f"作者: {', '.join([author.name for author in paper.authors])}\n")
            f.write(f"摘要: {paper.summary}\n")
            f.write(f"URL: {paper.entry_id}\n")
            f.write(f"图片输出目录: {images_dir}\n")
            f.write(f"LaTeX文件数量: {len(latex_files)}\n")
            f.write(f"图片文件总数: {len(all_images)}\n")
            f.write(f"匹配图片数量: {len(matched_images)}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("LaTeX文件列表:\n")
            for lf in latex_files:
                f.write(f"  - {lf}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("图片引用列表:\n")
            for ref in image_refs:
                f.write(f"  - {ref}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("匹配的图片文件:\n")
            for img in matched_images:
                f.write(f"  - {img}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("所有图片文件:\n")
            for img in all_images:
                f.write(f"  - {img}\n")

def main():
    extractor = ArxivLatexExtractor()
    conferences = {
        "cvpr": fetch_cvpr_titles,
        "iclr": fetch_iclr_titles,
        "neurips": fetch_neurips_titles,
        "icml": fetch_icml_titles,
    }
    
    year = 2024
    base_output_dir = 'all_conference_papers'
    for conf_name, fetch_func in conferences.items():
        print(f"\n=== {conf_name.upper()} ===")
        try:
            conf_output_dir = os.path.join(base_output_dir, conf_name)
            os.makedirs(conf_output_dir, exist_ok=True)

            accepted_titles = fetch_func(year)
            accepted_titles = list(accepted_titles)

            for title in tqdm.tqdm(accepted_titles[:3], desc=f"Downloading, cleaning and saving papers from {conf_name}"):
                try:
                    results = extractor.search_by_title(title)
                    if not results:
                        print(f"✗ 未找到文章: {title}")
                        continue

                    paper = results[0]
                    arxiv_id = paper.get_short_id()

                    article_output_dir = os.path.join(conf_output_dir, arxiv_id)
                    if os.path.exists(article_output_dir):
                        print(f"✓ 目录已存在，跳过: {arxiv_id}")
                        continue

                    result = extractor.process_article_by_title(title, article_output_dir)
                    if result:
                        print(f"✓ 成功处理: {result['paper'].title}")
                    else:
                        print(f"✗ 处理失败: {title}")
                except Exception as e:
                    print(f"✗ 处理文章时出错: {title} - {e}")
                    continue
        except Exception as e:
            print(f"处理会议 {conf_name} 时出现错误: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()