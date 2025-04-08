import os
import time
import json
import platform
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

class FirefoxBrowserBase:
    """
    Firefox浏览器基类，提供共享的浏览器功能
    """
    def __init__(self, headless=False, connect_existing=False, geckodriver_path=None):
        """
        初始化Firefox浏览器
        
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
        
        # 设置目录属性但不创建文件夹
        self.base_dir = "bilibili_data"
        self.subtitle_dir = os.path.join(self.base_dir, "subtitles")
        
        print("浏览器初始化完成")

    def save_json_data(self, data, directory, filename):
        """
        保存JSON数据到文件
        
        Args:
            data: 要保存的数据
            directory: 目录路径
            filename: 文件名
        
        Returns:
            保存的文件路径
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存: {filepath}")
        return filepath
    
    def load_json_data(self, json_path):
        """
        从JSON文件加载数据
        
        Args:
            json_path: JSON文件路径
        
        Returns:
            加载的数据
        """
        print(f"加载JSON文件: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"加载JSON文件时出错: {e}")
            return None
    
    def close(self):
        """
        关闭浏览器
        """
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
            print("浏览器已关闭") 