import os
import argparse
from pathlib import Path
import re

# 配置参数
MAX_CHARS_PER_FILE = 1000000  # 每个文件最大字数

def merge_documents(input_folder, output_folder):
    """合并文件夹中的文档文件"""
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有txt和md文件
    input_files = sorted([f for f in Path(input_folder).glob('*.*') if f.suffix.lower() in ['.txt', '.md']])
    
    if not input_files:
        print(f"在 {input_folder} 中没有找到文档文件")
        return
    
    current_content = []
    current_char_count = 0
    file_counter = 1
    table_of_contents = []
    
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 提取文件名中的评分信息（如果存在）
                score_match = re.search(r'(\d+)分_', file_path.stem)
                score = f"{score_match.group(1)}分 - " if score_match else ""
                
                # 使用评分和文件名作为章节标题
                title = f"## {score}{file_path.stem}\n\n"
                
                # 确保内容是标准的markdown格式
                if not content.startswith('```'):
                    content = f"```\n{content}\n```\n"
                
                content_with_title = title + content + "\n\n"
                content_length = len(content_with_title)
                
                # 检查是否需要创建新文件
                if current_char_count + content_length > MAX_CHARS_PER_FILE:
                    # 写入当前文件
                    output_file = os.path.join(output_folder, f'merged_doc_{file_counter}.md')
                    write_merged_file(output_file, current_content, table_of_contents)
                    
                    # 重置计数器和内容
                    file_counter += 1
                    current_content = []
                    table_of_contents = []
                    current_char_count = 0
                
                # 添加到目录
                display_name = f"{score}{file_path.stem}"
                table_of_contents.append(f"- [{display_name}](#)\n")
                
                # 添加内容
                current_content.append(content_with_title)
                current_char_count += content_length
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    # 写入最后一个文件
    if current_content:
        output_file = os.path.join(output_folder, f'merged_doc_{file_counter}.md')
        write_merged_file(output_file, current_content, table_of_contents)
        
    print(f"合并完成！共生成 {file_counter} 个文件")

def write_merged_file(output_file, content_list, toc_list):
    """写入合并后的文件，包含目录"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入文件标题
        f.write(f"# 评分文档合集\n\n")
        
        # 写入目录
        f.write("## 目录\n\n")
        f.writelines(toc_list)
        f.write("\n")
        
        # 写入正文内容
        f.writelines(content_list)

def main():
    parser = argparse.ArgumentParser(description='合并文档文件')
    parser.add_argument('--input_folder', type=str, default='scored_document',
                      help='输入文件夹路径，默认为"scored_document"')
    parser.add_argument('--output_folder', type=str, default='merged_documents',
                      help='输出文件夹路径，默认为"merged_documents"')
    
    args = parser.parse_args()
    
    # 获取绝对路径
    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)
    
    if not os.path.isdir(input_folder):
        print(f"错误: {input_folder} 不是有效的文件夹路径")
        return
    
    merge_documents(input_folder, output_folder)

if __name__ == '__main__':
    main()