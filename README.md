# Selenium Firefox自动化爬取工具

这是一个基于Selenium和Firefox的自动化网页内容爬取和处理工具。该工具主要用于自动化搜索、内容抓取、文档评分和合并等功能。

## 功能特点

- 自动化Google搜索
- 网页内容保存
- 文档智能评分
- 文档合并处理
- 支持无头模式运行
- 多线程处理能力
- API密钥轮换机制

## 项目结构

```
.
├── 1_google_search.py      # Google搜索模块
├── 2_web_content_saver.py  # 网页内容保存模块
├── 3_document_score.py     # 文档评分模块
├── 4_merge_documents.py    # 文档合并模块
├── config.py               # 配置文件
├── src/                    # 源代码目录
└── conda 环境搭建.md       # 环境配置说明
```

## 环境要求

- Python 3.x
- Firefox浏览器
- geckodriver
- 相关Python包（见环境配置说明）

## 安装说明

1. 克隆项目到本地
2. 按照`conda 环境搭建.md`中的说明配置环境
3. 确保已安装Firefox浏览器和对应版本的geckodriver
4. 在`config.py`中配置必要的API密钥

## 使用方法

### 1. Google搜索
在 搜索关键词.md 中添加需要搜索的关键词
```bash
python .\1_google_search.py
```

### 2. 保存网页内容

```bash
python .\2_web_content_saver.py
```

### 3. 文档评分

```bash
python .\3_document_score.py
```

### 4. 合并文档

```bash
python .\4_merge_documents.py
```

## 配置说明

在`config.py`中可以配置以下参数：

- API密钥列表
  API 密钥可以到 Gemini 官网申请
- 模型配置


