import os
import time
import hashlib
import re
from datetime import datetime
from urllib.parse import urlparse
from src.firefox_browser_base import FirefoxBrowserBase
from bs4 import BeautifulSoup
from pywinauto.application import Application
import pywinauto.timings
import pywinauto.findwindows
import pyperclip

class WebContentSaverBase(FirefoxBrowserBase):
    """
    基础网页内容保存器：用于访问URL并保存网页内容
    """
    def __init__(self, headless=False, connect_existing=True, geckodriver_path=None):
        """
        初始化Web内容保存器基类
        
        Args:
            headless: 是否使用无头模式（不显示浏览器窗口）
            connect_existing: 是否连接到已打开的Firefox浏览器
            geckodriver_path: geckodriver可执行文件的路径
        """
        super().__init__(headless=headless, connect_existing=connect_existing, geckodriver_path=geckodriver_path)
        
        # 创建内容保存的基础目录
        self.content_dir = "saved_content"
        os.makedirs(self.content_dir, exist_ok=True)
        
        print("Web内容保存器基类初始化完成")
    
    def sanitize_filename(self, filename):
        """
        清理文件名，移除不合法字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            清理后的文件名
        """
        # 移除非法文件名字符
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
        # 限制长度
        if len(sanitized) > 100:
            sanitized = sanitized[:97] + "..."
        return sanitized
    
    def get_domain(self, url):
        """
        从URL中提取域名
        
        Args:
            url: 完整URL
            
        Returns:
            域名字符串
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return domain
    
    def url_to_filename(self, url, title=""):
        """
        将URL转换为合适的文件名
        
        Args:
            url: 网页URL
            title: 网页标题
            
        Returns:
            适合作为文件名的字符串
        """
        domain = self.get_domain(url)
        
        # 使用标题和域名创建文件名
        if title:
            filename = f"{self.sanitize_filename(title)}__{domain}"
        else:
            # 如果没有标题，使用URL的一部分
            path = urlparse(url).path
            filename = f"{domain}{self.sanitize_filename(path)}"
            
        # 防止文件名过长
        if len(filename) > 100:
            # 使用URL的哈希值作为文件名一部分
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            if title:
                # 截取标题前部分 + 哈希 + 域名
                filename = f"{self.sanitize_filename(title[:50])}__{url_hash}__{domain}"
            else:
                filename = f"{domain}__{url_hash}"
        
        return filename
    
    def extract_text_from_html(self, html_content):
        """
        从HTML内容中提取纯文本内容
        
        Args:
            html_content: HTML内容字符串
            
        Returns:
            提取的纯文本内容
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除脚本和样式元素
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            
            # 获取文本
            text = soup.get_text()
            
            # 整理文本，删除多余空白行和空格
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            print(f"提取文本时出错: {str(e)}")
            return "文本提取失败: " + str(e)
    
    def save_webpage_content(self, url, title="", category=""):
        """
        访问URL并保存网页内容到本地
        
        Args:
            url: 要访问的网页URL
            title: 网页标题
            category: 分类目录名称
            
        Returns:
            bool: 是否成功保存
            str: 保存的文件路径或错误消息
        """
        try:
            print(f"正在访问: {url}")
            self.driver.get(url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 获取页面标题
            page_title = self.driver.title
            if not title:
                title = page_title
            
            # 获取页面内容
            page_source = self.driver.page_source
            
            # 创建保存目录（基于分类）
            if category:
                save_dir = os.path.join(self.content_dir, self.sanitize_filename(category))
            else:
                save_dir = self.content_dir
            
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            filename = self.url_to_filename(url, title)
            
            # 提取并保存纯文本内容
            text_content = self.extract_text_from_html(page_source)
            text_path = os.path.join(save_dir, f"{filename}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # 创建元数据（仅在内存中使用，不保存文件）
            metadata = {
                "url": url,
                "title": title,
                "page_title": page_title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "text_path": text_path
            }
            
            print(f"文本内容已保存: {text_path}")
            return True, text_path
            
        except Exception as e:
            error_msg = f"保存网页时出错: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def save_pdf(self, url, title="", category="", timeout=15):
        """
        访问PDF URL并点击保存按钮下载PDF，自动处理系统保存对话框
        主要使用键盘快捷键方式处理保存对话框，确保保存到项目目录
        
        Args:
            url: 要访问的PDF URL
            title: PDF标题（可选）
            category: 分类目录名称（可选）
            timeout: 等待PDF加载的超时时间（秒）
            
        Returns:
            bool: 是否成功保存
            str: 保存的文件路径或错误消息
        """
        # 创建保存目录（基于分类）
        if category:
            save_dir = os.path.join(self.content_dir, self.sanitize_filename(category))
        else:
            save_dir = self.content_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        if not title:
            # 从URL生成标题
            parsed_url = urlparse(url)
            title = os.path.basename(parsed_url.path)
        
        filename = self.url_to_filename(url, title)
        pdf_path = os.path.join(save_dir, f"{filename}.pdf")
        
        max_attempts = 2  # 最多尝试次数
        page_load_timeout = 20  # 设置页面加载超时时间（秒）
        
        for attempt in range(max_attempts):
            try:
                print(f"正在访问PDF链接: {url} (尝试 {attempt+1}/{max_attempts})")
                
                # 检查浏览器状态并在必要时重新初始化
                try:
                    # 尝试执行一个简单操作来验证浏览器是否仍然有效
                    self.driver.current_url
                except:
                    print("浏览器会话已失效，正在重新初始化...")
                    self.close()
                    time.sleep(2)
                    self.__init__(headless=self.headless, connect_existing=False, geckodriver_path=self.geckodriver_path)
                    time.sleep(2)
                
                # 设置页面加载超时
                self.driver.set_page_load_timeout(page_load_timeout)
                
                # 尝试访问URL，使用try-except捕获超时异常
                try:
                    # 访问URL
                    self.driver.get(url)
                except Exception as e:
                    print(f"页面加载超时 ({page_load_timeout}秒): {str(e)}")
                    if attempt < max_attempts - 1:
                        print(f"尝试下一次重试...")
                        continue
                    else:
                        return False, f"页面加载超时，无法访问PDF: {str(e)}"
                
                # 等待PDF加载
                print("等待PDF加载完成...")
                start_time = time.time()
                download_button = None
                check_interval = 0.5  # 检查间隔时间（秒）
                
                # 使用更短的检查周期，更快地检测到按钮或超时
                while time.time() - start_time < timeout:
                    # 检查是否已经超过10秒，如果是但仍没有进展，提前结束
                    if time.time() - start_time > 10 and download_button is None:
                        current_url = self.driver.current_url
                        if "pdf" not in current_url.lower():
                            print("10秒内未加载到PDF相关页面，提前终止")
                            break
                    
                    try:
                        # 尝试查找下载按钮
                        download_button = self.driver.find_element("id", "downloadButton")
                        if download_button and download_button.is_displayed():
                            break
                    except:
                        # 尝试其他可能的下载按钮 ID 或类
                        try:
                            download_button = self.driver.find_element("css selector", "[title='Download']")
                            if download_button and download_button.is_displayed():
                                break
                        except:
                            pass
                    
                    # 更短的等待间隔
                    time.sleep(check_interval)
                
                # 如果超时未找到下载按钮，直接使用快捷键保存
                if not download_button:
                    print("未找到下载按钮或加载超时，尝试使用浏览器快捷键保存...")
                    
                    # 使用Ctrl+S触发保存对话框
                    from pywinauto.keyboard import send_keys
                    send_keys("^s")
                    time.sleep(2)
                    
                    # 粘贴文件路径并保存
                    abs_pdf_path = os.path.abspath(pdf_path)
                    pyperclip.copy(abs_pdf_path)
                    time.sleep(1)
                    
                    # 粘贴完整路径 (Ctrl+V)
                    send_keys("^v")
                    time.sleep(1)
                    
                    # 按Enter键确认保存
                    send_keys("{ENTER}")
                    print("已使用键盘快捷键处理保存操作")
                    
                    # 等待保存完成
                    time.sleep(3)  # 减少等待时间
                    
                    if os.path.exists(pdf_path):
                        print(f"PDF已成功保存到: {pdf_path}")
                        return True, pdf_path
                
                # 如果找到下载按钮，执行点击下载
                if download_button:
                    print("点击PDF下载按钮...")
                    download_button.click()
                    
                    # 等待"另存为"对话框出现
                    print("等待系统'另存为'对话框出现...")
                    time.sleep(2)
                    
                    # 使用键盘快捷键处理保存对话框
                    try:
                        print("使用键盘快捷键处理对话框...")
                        from pywinauto.keyboard import send_keys
                        
                        # 粘贴完整路径
                        abs_pdf_path = os.path.abspath(pdf_path)
                        pyperclip.copy(abs_pdf_path)
                        
                        # 粘贴完整路径 (Ctrl+V)
                        send_keys("^v")
                        time.sleep(1)
                        
                        # 按Enter键确认保存
                        send_keys("{ENTER}")
                        print("已使用键盘快捷键处理保存操作")
                        
                        # 等待保存完成
                        time.sleep(3)  # 减少等待时间
                        
                        if os.path.exists(pdf_path):
                            print(f"PDF已成功保存到: {pdf_path}")
                            return True, pdf_path
                    except Exception as key_error:
                        print(f"使用键盘快捷键时出错: {key_error}")
                        if attempt < max_attempts - 1:
                            continue
                
                # 如果文件已经存在，视为成功
                if os.path.exists(pdf_path):
                    return True, pdf_path
                
                # 如果是最后一次尝试且失败，给用户时间手动处理
                if attempt == max_attempts - 1:
                    print("自动保存失败，可能需要手动操作")
                    print("等待5秒以便手动完成保存...")  # 减少等待时间
                    time.sleep(5)
                    
                    # 再次检查文件是否存在
                    if os.path.exists(pdf_path):
                        return True, pdf_path
                    
            except Exception as e:
                error_msg = f"保存PDF时出错 (尝试 {attempt+1}/{max_attempts}): {str(e)}"
                print(error_msg)
                
                if attempt < max_attempts - 1:
                    print(f"等待3秒后重试...")  # 减少等待时间
                    time.sleep(3)
                    continue
                else:
                    import traceback
                    traceback.print_exc()
                    return False, error_msg
        
        return False, "所有尝试都失败了，无法保存PDF" 