
1. **创建新的conda环境**
```bash
# 创建一个名为selenium_env的Python 3.10环境
conda create -n selenium_env python=3.10
```

2. **激活环境**
```bash
# 激活新创建的环境
conda activate selenium_env
```

3. **安装必要的依赖包**
```bash
pip install selenium webdriver-manager requests beautifulsoup4 pywinauto pyperclip
```

4. **安装Firefox浏览器**
- 访问 Mozilla Firefox 官网 (https://www.mozilla.org/firefox/) 下载并安装最新版本的Firefox浏览器

5. **安装Geckodriver**
- 访问 Geckodriver releases 页面 (https://github.com/mozilla/geckodriver/releases)
- 下载与您的操作系统匹配的版本
- 将下载的geckodriver解压并放置在以下位置之一：
  - Windows: 放在Python环境的Scripts目录下
  - Linux/Mac: 放在/usr/local/bin/目录下

6. **验证安装**
创建一个简单的测试脚本来验证环境是否正确配置：
```python
from selenium import webdriver
from selenium.webdriver.firefox.service import Service

# 初始化Firefox浏览器
driver = webdriver.Firefox()
driver.get("https://www.google.com")
print("Firefox浏览器启动成功！")
driver.quit()
```

如果你想操控已经打开的浏览器，参考：
https://blog.csdn.net/ohenix/article/details/139736741?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-2-139736741-blog-80390604.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-2-139736741-blog-80390604.235%5Ev43%5Epc_blog_bottom_relevance_base9&utm_relevant_index=5