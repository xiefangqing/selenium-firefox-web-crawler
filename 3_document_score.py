import os
import json
import time
import requests
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# 移除原有配置项
from config import API_KEYS, MAX_RETRIES, RETRY_DELAY, MAX_WORKERS
# 新增 Gemini 特定配置
from config import GEMINI_MODEL, GEMINI_API_URL

def mask_api_key(api_key):
    """遮蔽API密钥中间部分，只显示前4位和后4位"""
    if len(api_key) <= 8:
        return "****"
    return api_key[:4] + "****" + api_key[-4:]

class APIManager:
    """API调用管理器，处理密钥轮换和限流"""
    
    def __init__(self, api_keys, model):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model = model
        # 添加每个密钥的最后使用时间和冷却状态跟踪
        self.last_used_time = defaultdict(float)
        self.cooling_keys = set()
        # 添加默认冷却时间（秒）
        self.cooling_period = 60
        
    def get_current_key(self):
        """获取当前使用的API密钥"""
        return self.api_keys[self.current_key_index]
    
    def rotate_key(self, cooling=False):
        """轮换到下一个可用的API密钥"""
        current_key = self.get_current_key()
        
        # 如果当前密钥需要冷却，将其标记
        if cooling:
            self.cooling_keys.add(current_key)
            print(f"API密钥 {mask_api_key(current_key)} 进入冷却期 {self.cooling_period} 秒")
            # 设置冷却结束时间
            self.last_used_time[current_key] = time.time()
        
        # 查找下一个未在冷却中的密钥
        original_index = self.current_key_index
        while True:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            next_key = self.get_current_key()
            
            # 检查密钥是否在冷却中
            if next_key in self.cooling_keys:
                # 检查冷却时间是否已过
                elapsed = time.time() - self.last_used_time[next_key]
                if elapsed >= self.cooling_period:
                    # 冷却时间已过，移除冷却标记
                    self.cooling_keys.remove(next_key)
                    print(f"API密钥 {mask_api_key(next_key)} 冷却结束，可以使用")
                    break
            else:
                # 密钥未在冷却中，可以使用
                break
                
            # 如果已经检查了所有密钥但都在冷却中
            if self.current_key_index == original_index:
                # 找出冷却结束最早的密钥
                min_wait = float('inf')
                earliest_key = None
                
                for key in self.cooling_keys:
                    remaining = self.cooling_period - (time.time() - self.last_used_time[key])
                    if remaining < min_wait:
                        min_wait = remaining
                        earliest_key = key
                
                if min_wait > 0:
                    print(f"所有API密钥都在冷却中，等待 {min_wait:.2f} 秒后重试")
                    time.sleep(min_wait + 1)  # 额外等待1秒以确保冷却完成
                
                # 找到冷却结束的密钥索引
                self.current_key_index = self.api_keys.index(earliest_key)
                self.cooling_keys.remove(earliest_key)
                break
        
        print(f"轮换到下一个API密钥: {mask_api_key(self.get_current_key())}")
        
    def call_api(self, messages):
        """调用 Gemini API 并处理响应"""
        api_key = self.get_current_key()
        
        # 记录本次使用时间
        self.last_used_time[api_key] = time.time()
        
        # 构建请求参数 - Gemini API 格式
        # 将消息格式转换为 Gemini 格式
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            }
        }
        
        # 构建 API URL (添加 API 密钥作为查询参数)
        full_url = f"{GEMINI_API_URL}/{self.model}:generateContent?key={api_key}"
        
        # 请求头
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 发送请求
        try:
            response = requests.post(full_url, headers=headers, json=payload)
            
            # 检查响应状态
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:  # 请求格式错误
                print(f"请求格式错误，状态码: {response.status_code}，响应: {response.text}")
                return None
            elif response.status_code == 401:  # 认证错误
                print(f"API密钥认证错误，状态码: {response.status_code}，响应: {response.text}")
                self.rotate_key()
                return None
            elif response.status_code == 429:  # 请求过多，切换API密钥并进入冷却期
                print(f"请求频率限制，状态码: {response.status_code}")
                self.rotate_key(cooling=True)
                return None
            else:  # 其他错误
                print(f"请求失败，状态码: {response.status_code}，响应: {response.text}")
                return None
        except Exception as e:
            print(f"请求异常: {str(e)}")
            return None
    
    def call_api_with_retry(self, messages, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
        """调用API并在失败时重试"""
        retries = 0
        while retries <= max_retries:
            response = self.call_api(messages)
            if response:
                return response
            else:
                retries += 1
                if retries <= max_retries:
                    delay = retry_delay * (2 ** (retries - 1))  # 指数退避
                    print(f"第 {retries} 次重试，等待 {delay} 秒...")
                    time.sleep(delay)
                else:
                    print(f"超过最大重试次数 {max_retries}，放弃请求")
                    return None

def score_document(api_manager, file_path, input_folder, output_folder):
    """对文档进行评分"""
    try:
        # 获取文件名（现在直接使用基本文件名，不保留子目录结构）
        file_name = os.path.basename(file_path)
        
        # 如果需要考虑来自不同子目录的同名文件，可以在文件名中添加相对路径的哈希
        if os.path.dirname(os.path.relpath(file_path, input_folder)):
            rel_path = os.path.relpath(file_path, input_folder)
            dir_part = os.path.dirname(rel_path).replace('\\', '_').replace('/', '_')
            file_name = f"{dir_part}_{file_name}"
        
        # 检查输出文件是否已存在
        existing_files = [f for f in os.listdir(output_folder) if f.split('分_')[-1] == file_name]
        if existing_files:
            print(f"文件已处理，跳过: {file_path}")
            return True
            
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 构建提示
        messages = [
            {"role": "system", "content": "你是一个专业的强化学习和深度Q网络(DQN)教学内容评估专家，精通强化学习理论、DQN算法原理、实现细节和应用场景。你将对文档进行全面评分，按照指定标准提供详细评分理由并总结文档教学价值。"},
            {"role": "user", "content": f"""作为强化学习和DQN专家，请你对以下文档进行全面评分，基于以下标准进行1-5分的评分，提供详细评分理由并总结文档教学价值：

## 评分标准 (各项满分5分):

1. **核心概念准确性 (Accuracy)**：内容在解释DQN核心概念（如Q-learning基础、神经网络近似、经验回放、目标网络等）时是否科学、严谨、无误。

2. **解释清晰易懂度 (Clarity)**：内容的语言表达是否清晰、流畅，是否易于目标学习者理解。是否有效使用了比喻、类比或简化解释。

3. **内容结构逻辑性 (Structure)**：内容的组织结构是否合理，逻辑递进关系是否清晰。从基础到深入，或者从理论到实践的过渡是否自然。

4. **原理深度与启发性 (Depth & Insight)**：内容是否不仅陈述事实，而是能解释清楚"为什么"这么设计，是否能引发学习者对原理的深入思考。

5. **实现细节实用性 (Practicality)**：关于实现、代码示例、超参数设置等内容是否具有实践指导意义，是否对学习者动手实现DQN有帮助。

6. **案例/应用相关性与说明性 (Relevance & Illustration)**：所引用的案例或应用领域是否与DQN紧密相关，并且能有效说明DQN的能力、特点或局限性。

7. **教学内容适宜性 (Suitability for Teaching)**：综合考虑以上各点，评估该部分内容是否适合直接或经过少量修改后用于教学文档或PPT制作。

## 输出格式:

```
### [文档标题]

#### 评分:
- 核心概念准确性: [1-5分] - [简短理由]
- 解释清晰易懂度: [1-5分] - [简短理由]
- 内容结构逻辑性: [1-5分] - [简短理由]
- 原理深度与启发性: [1-5分] - [简短理由]
- 实现细节实用性: [1-5分] - [简短理由]
- 案例/应用相关性: [1-5分] - [简短理由]
- 教学内容适宜性: [1-5分] - [简短理由]
- **总分**: [总分/35] ([百分比])

#### 评价摘要:
[2-3句话总结文档的教学价值、优势和不足]

```

以下是需要评分的文档内容：

{content}"""}
        ]
        
        # 调用API
        response = api_manager.call_api_with_retry(messages)
        
        # 解析 Gemini API 响应
        if response and "candidates" in response and len(response["candidates"]) > 0:
            # 获取生成的内容
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    scoring_result = parts[0]["text"]
                    
                    # 提取总分
                    total_score_match = re.search(r'\*\*总分\*\*:\s*(\d+)/35', scoring_result)
                    if total_score_match:
                        total_score = int(total_score_match.group(1))
                        
                        # 在原始文件名前添加评分
                        new_file_name = f"{total_score}分_{file_name}"
                        new_output_path = os.path.join(output_folder, new_file_name)
                        
                        # 保存评分结果到文件
                        with open(new_output_path, 'w', encoding='utf-8') as f:
                            f.write(f"```\n{scoring_result}\n```\n\n{content}")
                        
                        print(f"已成功评分并保存文档: {new_output_path}")
                        return True
                    else:
                        print(f"无法从评分结果中提取总分: {file_path}")
                        
                        # 保存原始评分结果，以便进行检查
                        debug_path = os.path.join(output_folder, f"debug_{os.path.basename(file_path)}")
                        with open(debug_path, 'w', encoding='utf-8') as f:
                            f.write(scoring_result)
                        
                        return False
        
        print(f"处理文档失败: {file_path}")
        return False
    
    except Exception as e:
        print(f"处理文档 {file_path} 时出错: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """处理指定文件夹中的所有文档"""
    # 创建API管理器，使用 Gemini 模型
    api_manager = APIManager(API_KEYS, GEMINI_MODEL)
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有待处理的文本文件 (.txt和.md格式)
    files_to_process = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.txt', '.md', '.pdf')):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)
                
                # 如果来自子目录，添加目录信息到文件名以避免冲突
                if os.path.dirname(os.path.relpath(file_path, input_folder)):
                    rel_path = os.path.relpath(file_path, input_folder)
                    dir_part = os.path.dirname(rel_path).replace('\\', '_').replace('/', '_')
                    file_name = f"{dir_part}_{file_name}"
                
                # 检查输出目录中是否已有处理过的文件（以数字分数开头）
                existing_files = [f for f in os.listdir(output_folder) if f.split('分_')[-1] == file_name]
                if not existing_files:
                    files_to_process.append(file_path)
                    print(f"添加待处理文件: {file_path}")
                else:
                    print(f"文件已处理，跳过: {file_path}")
    
    print(f"找到 {len(files_to_process)} 个文件待处理")
    
    # 使用线程池并行处理文件，但减少并发数以避免频率限制
    success_count = 0
    # 根据API密钥数量动态调整并发数，确保每个密钥不会过载
    effective_workers = min(MAX_WORKERS, max(1, len(API_KEYS)))
    print(f"使用 {effective_workers} 个工作线程处理文件")
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(score_document, api_manager, file_path, input_folder, output_folder): file_path 
            for file_path in files_to_process
        }
        
        # 处理结果
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"处理 {file_path} 时发生异常: {str(e)}")
    
    print(f"处理完成: {success_count}/{len(files_to_process)} 个文件成功评分")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对文档进行评分')
    parser.add_argument('--input_folder', type=str, default='saved_content', help='要处理的文件夹路径，默认为"saved_content"')
    parser.add_argument('--output_folder', type=str, default='scored_document', help='输出文件夹路径，默认为"scored_document"')
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"错误: {args.input_folder} 不是有效的文件夹路径")
    else:
        process_folder(args.input_folder, args.output_folder)