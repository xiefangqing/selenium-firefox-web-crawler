import os
import time
import platform
import json
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

class GoogleSearcher:
    """
    Google搜索工具，用于在Google上搜索关键词
    """
    def __init__(self, headless=False, connect_existing=True, geckodriver_path=None):
        """
        初始化Google搜索工具
        
        Args:
            headless: 是否使用无头模式（不显示浏览器窗口）
            connect_existing: 是否连接到已打开的Firefox浏览器
            geckodriver_path: geckodriver可执行文件的路径
        """
        print("初始化Firefox浏览器...")
        
        # 如果未指定geckodriver路径，根据操作系统设置默认路径
        if geckodriver_path is None:
            if platform.system() == "Windows":
                geckodriver_path = "geckodriver.exe"  # Windows上默认添加.exe扩展名
            else:
                geckodriver_path = "geckodriver"
        
        if connect_existing:
            try:
                # 连接到已打开的Firefox浏览器，使用Marionette协议
                print(f"尝试使用geckodriver路径: {geckodriver_path}")
                
                # 检查文件是否存在
                if not os.path.isfile(geckodriver_path):
                    raise FileNotFoundError(f"找不到geckodriver可执行文件: {geckodriver_path}")
                
                service = Service(executable_path=geckodriver_path)
                service.service_args = ['--marionette-port', '2828', '--connect-existing']
                
                # 使用已打开的浏览器会话
                self.driver = webdriver.Firefox(service=service)
                print("已连接到运行中的Firefox浏览器")
            except (FileNotFoundError, WebDriverException) as e:
                print(f"连接到已打开的Firefox浏览器失败: {str(e)}")
                print("请确保geckodriver可执行文件存在且Firefox浏览器已打开")
                print("尝试启动新的Firefox浏览器实例...")
                
                # 启动新的Firefox浏览器实例
                options = Options()
                if headless:
                    options.add_argument('-headless')  # 无头模式
                self.driver = webdriver.Firefox(options=options)
        else:
            # 启动新的Firefox浏览器实例
            options = Options()
            if headless:
                options.add_argument('-headless')  # 无头模式
            
            self.driver = webdriver.Firefox(options=options)
        
        self.wait = WebDriverWait(self.driver, 10)
        print("浏览器初始化完成")
        
        # 创建保存结果的目录
        self.base_dir = "google_data"
        self.results_dir = os.path.join(self.base_dir, "search_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def search(self, keyword):
        """
        在Google上搜索关键词
        
        Args:
            keyword: 要搜索的关键词
            
        Returns:
            dict: 搜索结果数据，包含搜索结果
        """
        try:
            print(f"正在打开Google搜索页面...")
            self.driver.get("https://www.google.com")
            
            # 等待Google页面加载
            search_box = self.wait.until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            print("Google页面加载完成")
            
            # 输入搜索关键词
            print(f"正在搜索关键词: {keyword}")
            search_box.clear()
            search_box.send_keys(keyword)
            search_box.send_keys(Keys.RETURN)
            
            # 等待搜索结果加载
            self.wait.until(
                EC.presence_of_element_located((By.ID, "search"))
            )
            print("搜索结果加载完成")
            
            # 获取搜索结果统计
            result_stats_text = ""
            try:
                result_stats = self.driver.find_element(By.ID, "result-stats")
                result_stats_text = result_stats.text
                print(f"搜索结果: {result_stats_text}")
            except NoSuchElementException:
                print("无法获取搜索结果统计")
            
            # 获取搜索结果列表 - 使用多种可能的选择器
            search_results = []
            selectors = [
                "#search .g", 
                ".MjjYud", 
                ".Gx5Zad", 
                ".tF2Cxc",
                "div[data-hveid]",
                ".kb0PBd",
                "[data-snc]"
            ]
            
            for selector in selectors:
                try:
                    result_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if result_elements:
                        print(f"找到 {len(result_elements)} 个搜索结果，使用选择器: {selector}")
                        
                        for i, result in enumerate(result_elements):
                            try:
                                # 尝试多种选择器提取标题和链接
                                title = ""
                                link = ""
                                description = ""
                                
                                # 尝试提取标题
                                title_selectors = ["h3", ".LC20lb", ".DKV0Md", ".vvjwJb"]
                                for title_selector in title_selectors:
                                    try:
                                        title_element = result.find_element(By.CSS_SELECTOR, title_selector)
                                        title = title_element.text
                                        if title:
                                            break
                                    except:
                                        continue
                                
                                # 尝试提取链接
                                link_selectors = ["a", "a[href]", ".yuRUbf a", ".kvH3mc a"]
                                for link_selector in link_selectors:
                                    try:
                                        link_element = result.find_element(By.CSS_SELECTOR, link_selector)
                                        link = link_element.get_attribute("href")
                                        if link:
                                            break
                                    except:
                                        continue
                                
                                # 尝试提取描述
                                desc_selectors = [".VwiC3b", ".yXK7lf", ".MUxGbd", ".lyLwlc"]
                                for desc_selector in desc_selectors:
                                    try:
                                        desc_element = result.find_element(By.CSS_SELECTOR, desc_selector)
                                        description = desc_element.text
                                        if description:
                                            break
                                    except:
                                        continue
                                
                                # 只添加有标题或链接的结果
                                if title or link:
                                    search_results.append({
                                        "index": i + 1,
                                        "title": title,
                                        "link": link,
                                        "description": description
                                    })
                                    
                            except Exception as e:
                                print(f"提取搜索结果 #{i+1} 时出错: {e}")
                        
                        # 如果找到了结果，就跳出循环
                        if search_results:
                            break
                except Exception as e:
                    print(f"使用选择器 '{selector}' 提取搜索结果时出错: {e}")
            
            # 如果所有选择器都失败，尝试通过页面源代码提取搜索结果
            if not search_results:
                print("尝试从页面源代码提取搜索结果...")
                html_source = self.driver.page_source
                
                # 提取所有链接和标题
                link_pattern = r'<a href="(https?://[^"]+)"[^>]*>(.*?)</a>'
                matches = re.findall(link_pattern, html_source)
                
                for i, (link, title_html) in enumerate(matches):
                    # 过滤掉Google的内部链接
                    if "google.com" not in link and link.startswith("http"):
                        # 清除HTML标签
                        title = re.sub(r'<[^>]+>', '', title_html).strip()
                        if title:
                            search_results.append({
                                "index": i + 1,
                                "title": title,
                                "link": link,
                                "description": ""
                            })
                
                if search_results:
                    print(f"从页面源代码中提取到 {len(search_results)} 个搜索结果")
            
            print(f"总共找到 {len(search_results)} 个搜索结果")
            
            # 保存搜索结果
            search_data = {
                "keyword": keyword,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stats": result_stats_text,
                "results_count": len(search_results),
                "results": search_results
            }
            
            # 保存搜索结果
            self.save_search_results(keyword, search_data)
            
            return search_data
            
        except TimeoutException:
            print("页面加载超时")
            return {"error": "页面加载超时", "keyword": keyword}
        except Exception as e:
            print(f"搜索过程中出错: {e}")
            return {"error": str(e), "keyword": keyword}
    
    def save_search_results(self, keyword, search_data):
        """
        保存搜索结果到JSON文件
        
        Args:
            keyword: 搜索关键词
            search_data: 搜索结果数据
        """
        # 生成文件名，使用时间戳避免重名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{keyword}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # 保存为JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, ensure_ascii=False, indent=2)
        
        print(f"搜索结果已保存到: {filepath}")
    
    def save_json_data(self, data, directory, filename):
        """
        保存JSON数据到文件
        
        Args:
            data: 要保存的数据
            directory: 保存目录
            filename: 文件名
        """
        # 确保目录存在
        os.makedirs(directory, exist_ok=True)
        
        # 保存JSON数据
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {filepath}")
    
    def search_multiple(self, keywords, save_summary=False):
        """
        搜索多个关键词
        
        Args:
            keywords: 关键词列表
            save_summary: 是否保存搜索汇总结果，默认为False
            
        Returns:
            dict: 搜索结果统计
        """
        results = {
            "total": len(keywords),
            "success": 0,
            "failed": 0,
            "searches": []
        }
        
        for i, keyword in enumerate(keywords):
            print(f"\n[{i+1}/{len(keywords)}] 搜索关键词: {keyword}")
            
            search_data = self.search(keyword)
            
            # 更新统计
            if "error" not in search_data:
                results["success"] += 1
                search_status = "成功"
            else:
                results["failed"] += 1
                search_status = "失败"
            
            # 添加到搜索结果列表
            results["searches"].append({
                "keyword": keyword,
                "status": search_status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results_count": search_data.get("results_count", 0)
            })
            
            # 在搜索之间添加延迟
            if i < len(keywords) - 1:
                delay = 2
                print(f"等待 {delay} 秒后继续...")
                time.sleep(delay)
        
        print(f"\n搜索完成!")
        print(f"总关键词数: {results['total']}")
        print(f"成功: {results['success']}")
        print(f"失败: {results['failed']}")
        
        # 只有当save_summary为True时才保存汇总结果
        if save_summary:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"搜索汇总_{timestamp}.json"
            self.save_json_data(results, self.results_dir, filename)
            print(f"搜索汇总已保存到: {filename}")
        else:
            print("根据设置，未保存搜索汇总结果")
        
        return results
    
    def close(self):
        """
        关闭浏览器
        """
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
            print("浏览器已关闭")

# 新增: 从Markdown文件读取关键词
def read_keywords_from_markdown(file_path):
    """
    从Markdown文件中读取关键词列表
    
    Args:
        file_path: Markdown文件路径
        
    Returns:
        list: 关键词列表
    """
    try:
        print(f"正在从文件读取关键词: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 过滤空行，并去除每行前后的空白字符
        keywords = [line.strip() for line in lines if line.strip()]
        print(f"已读取 {len(keywords)} 个关键词")
        return keywords
    except Exception as e:
        print(f"读取关键词文件时出错: {e}")
        return []

# 使用示例
if __name__ == "__main__":
    import sys
    
    # 新增: 处理文件输入参数
    keywords = []
    file_path = None
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果第一个参数是文件路径，则从文件读取关键词
        if os.path.isfile(sys.argv[1]):
            file_path = sys.argv[1]
            keywords = read_keywords_from_markdown(file_path)
        else:
            # 否则将所有参数作为关键词
            keywords = sys.argv[1:]
    
    # 如果没有命令行参数或文件中没有关键词，使用默认关键词
    if not keywords:
        # 尝试从默认文件读取
        default_file = "搜索关键词.md"
        if os.path.isfile(default_file):
            keywords = read_keywords_from_markdown(default_file)
        
        # 如果仍然没有关键词，使用硬编码的默认值
        if not keywords:
            keywords = ["Python Selenium", "Web Automation"]
            print(f"未指定关键词，使用默认关键词: {keywords}")
    
    # 使用固定的geckodriver路径
    geckodriver_path = r"C:\python3\geckodriver.exe"
    if not os.path.exists(geckodriver_path):
        geckodriver_path = "geckodriver.exe"  # 回退到默认位置
    
    print(f"使用geckodriver路径: {geckodriver_path}")
    
    # 创建Google搜索器实例
    searcher = GoogleSearcher(
        headless=False, 
        connect_existing=True,  # 尝试连接到已打开的Firefox浏览器
        geckodriver_path=geckodriver_path
    )
    
    try:
        # 执行搜索，设置save_summary=False不保存汇总
        searcher.search_multiple(keywords, save_summary=False)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭浏览器
        searcher.close() 