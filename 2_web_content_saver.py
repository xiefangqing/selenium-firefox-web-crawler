import os
import time
import json
from datetime import datetime
from web_content_saver_base import WebContentSaverBase

class WebContentSaver(WebContentSaverBase):
    """
    用于处理搜索结果并保存网页内容的工具
    """
    def __init__(self, headless=False, connect_existing=True, geckodriver_path=None):
        """
        初始化Web内容处理器
        
        Args:
            headless: 是否使用无头模式（不显示浏览器窗口）
            connect_existing: 是否连接到已打开的Firefox浏览器
            geckodriver_path: geckodriver可执行文件的路径
        """
        super().__init__(headless=headless, connect_existing=connect_existing, geckodriver_path=geckodriver_path)
        print("搜索结果处理器初始化完成")
    
    def is_pdf_link(self, url):
        """
        判断URL是否是PDF链接
        
        Args:
            url: 要检查的URL
            
        Returns:
            bool: 是否是PDF链接
        """
        # 检查URL是否直接以.pdf结尾
        if url.lower().endswith('.pdf'):
            return True
        
        # 检查URL中是否包含pdf参数或路径
        if '.pdf' in url.lower():
            return True
            
        return False
    
    def process_search_result_file(self, json_file_path, max_links=10, delay=2):
        """
        处理单个搜索结果JSON文件
        
        Args:
            json_file_path: JSON文件路径
            max_links: 最多处理的链接数量
            delay: 请求之间的延迟（秒）
            
        Returns:
            dict: 处理结果统计
        """
        print(f"\n开始处理搜索结果文件: {json_file_path}")
        
        # 加载JSON文件
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                search_data = json.load(f)
        except Exception as e:
            print(f"无法加载JSON文件: {str(e)}")
            return {"error": str(e), "processed": 0, "success": 0, "failed": 0}
        
        # 获取搜索关键词
        keyword = search_data.get("keyword", "unknown")
        print(f"搜索关键词: {keyword}")
        
        # 获取搜索结果列表
        results = search_data.get("results", [])
        print(f"找到 {len(results)} 个搜索结果")
        
        # 处理统计
        stats = {
            "keyword": keyword,
            "total": min(len(results), max_links),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        # 处理每个搜索结果
        for i, result in enumerate(results):
            # 限制处理的链接数量
            if i >= max_links:
                remaining = len(results) - max_links
                print(f"已达到最大链接处理数量 ({max_links})，跳过剩余 {remaining} 个链接")
                stats["skipped"] = remaining
                break
            
            link = result.get("link", "")
            title = result.get("title", "")
            
            if not link or link.startswith("https://www.google.com"):
                print(f"跳过无效链接或Google内部链接: {link}")
                stats["skipped"] += 1
                continue
            
            print(f"\n[{i+1}/{min(len(results), max_links)}] 处理链接: {link}")
            stats["processed"] += 1
            
            # 检查浏览器状态
            try:
                # 尝试执行一个简单操作来验证浏览器是否仍然有效
                self.driver.current_url
            except Exception as e:
                print(f"浏览器会话已失效，正在重新初始化... 错误: {str(e)}")
                try:
                    self.close()
                    time.sleep(2)
                    super().__init__(headless=self.headless, connect_existing=False, geckodriver_path=self.geckodriver_path)
                    time.sleep(2)
                    print("浏览器已成功重新初始化")
                except Exception as e2:
                    print(f"重新初始化浏览器失败: {str(e2)}")
                    stats["failed"] += 1
                    continue
            
            # 检查是否是PDF链接
            if self.is_pdf_link(link):
                print(f"检测到PDF链接，使用PDF保存方法")
                success, result_path = self.save_pdf(link, title, keyword)
            else:
                # 保存普通网页内容
                success, result_path = self.save_webpage_content(link, title, keyword)
            
            # 更新统计
            if success:
                stats["success"] += 1
                status = "成功"
            else:
                stats["failed"] += 1
                status = "失败"
            
            # 记录详情
            stats["details"].append({
                "url": link,
                "title": title,
                "status": status,
                "result": result_path
            })
            
            # 请求之间添加延迟
            if i < min(len(results), max_links) - 1:
                print(f"等待 {delay} 秒后继续...")
                time.sleep(delay)
        
        print(f"\n文件处理完成: {json_file_path}")
        print(f"总链接数: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        print(f"跳过: {stats['skipped']}")
        
        return stats
    
    def process_search_results_folder(self, folder_path=None, max_files=None, max_links_per_file=5, delay=2):
        """
        处理文件夹中的所有搜索结果JSON文件
        
        Args:
            folder_path: 包含搜索结果JSON文件的文件夹路径，默认为google_data/search_results
            max_files: 最多处理的文件数量，默认为None（处理所有文件）
            max_links_per_file: 每个文件最多处理的链接数量
            delay: 请求之间的延迟（秒）
            
        Returns:
            dict: 处理结果统计
        """
        # 设置默认文件夹路径
        if folder_path is None:
            folder_path = os.path.join("google_data", "search_results")
        
        print(f"\n开始处理搜索结果文件夹: {folder_path}")
        
        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            # 尝试备选路径
            alt_folder_path = "search_results"
            if os.path.isdir(alt_folder_path):
                folder_path = alt_folder_path
                print(f"使用备选路径: {folder_path}")
            else:
                error_msg = f"文件夹不存在: {folder_path} 或 {alt_folder_path}"
                print(error_msg)
                return {"error": error_msg}
        
        # 获取所有JSON文件
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 限制处理的文件数量
        if max_files and len(json_files) > max_files:
            print(f"将只处理前 {max_files} 个文件")
            json_files = json_files[:max_files]
        
        # 处理统计
        overall_stats = {
            "total_files": len(json_files),
            "processed_files": 0,
            "total_links": 0,
            "successful_links": 0,
            "failed_links": 0,
            "skipped_links": 0,
            "file_results": []
        }
        
        # 处理每个JSON文件
        for i, json_file in enumerate(json_files):
            file_path = os.path.join(folder_path, json_file)
            print(f"\n[{i+1}/{len(json_files)}] 处理文件: {file_path}")
            
            # 处理单个文件
            file_stats = self.process_search_result_file(
                file_path, 
                max_links=max_links_per_file, 
                delay=delay
            )
            
            # 更新总体统计
            overall_stats["processed_files"] += 1
            overall_stats["total_links"] += file_stats.get("total", 0)
            overall_stats["successful_links"] += file_stats.get("success", 0)
            overall_stats["failed_links"] += file_stats.get("failed", 0)
            overall_stats["skipped_links"] += file_stats.get("skipped", 0)
            overall_stats["file_results"].append({
                "file": json_file,
                "keyword": file_stats.get("keyword", "unknown"),
                "stats": file_stats
            })
            
            # 在处理文件之间添加更长的延迟
            if i < len(json_files) - 1:
                file_delay = delay * 2
                print(f"等待 {file_delay} 秒后继续下一个文件...")
                time.sleep(file_delay)
        
        print("\n所有文件处理完成")
        print(f"处理的文件数: {overall_stats['processed_files']}")
        print(f"总链接数: {overall_stats['total_links']}")
        print(f"成功: {overall_stats['successful_links']}")
        print(f"失败: {overall_stats['failed_links']}")
        print(f"跳过: {overall_stats['skipped_links']}")
        
        # 保存总体统计到JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_filename = f"content_download_stats_{timestamp}.json"
        stats_path = os.path.join(self.content_dir, stats_filename)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计结果已保存到: {stats_path}")
        
        return overall_stats


# 使用示例
if __name__ == "__main__":
    import sys
    
    # 默认参数
    max_files = None  # 默认处理所有文件
    max_links_per_file = 5  # 每个文件最多处理5个链接
    delay = 3  # 请求之间的延迟（秒）
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        try:
            max_files = int(sys.argv[1])
            print(f"设置最大处理文件数为: {max_files}")
        except ValueError:
            print(f"无效的最大文件数参数: {sys.argv[1]}，使用默认值: 所有文件")
    
    if len(sys.argv) > 2:
        try:
            max_links_per_file = int(sys.argv[2])
            print(f"设置每个文件最大处理链接数为: {max_links_per_file}")
        except ValueError:
            print(f"无效的每个文件最大链接数参数: {sys.argv[2]}，使用默认值: 5")
    
    if len(sys.argv) > 3:
        try:
            delay = int(sys.argv[3])
            print(f"设置请求延迟为: {delay}秒")
        except ValueError:
            print(f"无效的延迟参数: {sys.argv[3]}，使用默认值: 3秒")
    
    # 使用固定的geckodriver路径
    geckodriver_path = r"C:\python3\geckodriver.exe"
    if not os.path.exists(geckodriver_path):
        geckodriver_path = "geckodriver.exe"  # 回退到默认位置
    
    print(f"使用geckodriver路径: {geckodriver_path}")
    
    # 创建内容保存器实例
    content_saver = WebContentSaver(
        headless=False,
        connect_existing=True,  # 尝试连接到已打开的Firefox浏览器
        geckodriver_path=geckodriver_path
    )
    
    try:
        # 处理搜索结果文件夹
        content_saver.process_search_results_folder(
            max_files=max_files,
            max_links_per_file=max_links_per_file,
            delay=delay
        )
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭浏览器
        content_saver.close() 