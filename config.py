
OUTPUT_DIR = "research_results" 

# API 配置
API_KEYS = [
    "AIzaSyAvxxLKHWXuRQMXFIGwqv7AzzYne9xib68",

]
MODEL = "gemini-2.0-flash"
API_URL = "https://www.dmxapi.com/v1/chat/completions"  # 替换为实际API端点

# 运行参数
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒
MAX_WORKERS = 4  # 并行处理线程数

# Gemini API 配置
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models"