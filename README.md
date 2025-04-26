# Selenium Firefox 自动化爬取工具

这是一个基于 Selenium 和 Firefox 的自动化网页内容爬取和处理工具。该工具主要用于自动化搜索、内容抓取、文档评分和合并等功能。

## 功能特点

-   自动化 Google 搜索
-   网页内容保存
-   文档智能评分
-   文档合并处理
-   支持无头模式运行
-   多线程处理能力
-   API 密钥轮换机制

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

-   Python 3.x
-   Firefox 浏览器
-   geckodriver
-   相关 Python 包（见环境配置说明）

## 安装说明

1. 克隆项目到本地
2. 按照`conda 环境搭建.md`中的说明配置环境
3. 确保已安装 Firefox 浏览器和对应版本的 geckodriver
4. 在`config.py`中配置必要的 API 密钥

## 使用方法

### 1. Google 搜索

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

-   API 密钥列表
    API 密钥可以到 Gemini 官网申请
-   模型配置

## 使用流程

1.  通过[AutoGLM 沉思]产出研究报告，作为基础资料(A)

    比如：我想通过制作 PPT 来深入了解 DQN 强化学习算法，帮我整理尽可能全面的内容

2.  使用大模型：参考(A)，给出一份评分体系用于筛选高质量内容

    比如：参考这份文档，我想通过筛选关于 DQN 强化学习方面高质量内容来整合产出一份教学文档并做成 PPT，给我对应的评分系统让我筛选内容

3.  尽可能全面的爬取全网信息，让大模型推荐一些搜索关键词，中文和英文都提供

        比如：

        1. 我想通过尽可能多的搜集内容来制作 DQN 强化学习的教程，给我推荐一些搜索关键词
        2. 英文的关键词也提供一些
        3. 只保留关键词，比如：强化学习入门\n Q-learning 详解\n Deep Q-Network (DQN)\n Q-learning

4.  依次执行 1_google_search.py 2_web_content_saver.py，爬取内容

5.  考虑到大模型的上下文是有限的，利用评分体系给爬取内容打分，将没有价值的内容去除，即高分保留低分去除

    比如：参考评分系统 @/评分系统/DQN 教学内容质量评分系统.md ，修改 @/3_document_score.py 代码中的提示词，将 @/saved_content/ 中的所有文件使用大模型打分

6.  执行 4_merge_documents.py，将所有优质内容合并为一份文档

7.  最后，把合并好的文档扔给大模型（需要上下文够大，比如 Gemini）作为参考，就能得到高质量的回答

    比如：参考文档中的内容，我的需求是我想通过制作 PPT 教程的方式来深入了解 DQN 强化学习算法，给出教程，每张 PPT 提供对应的解说词，对于复杂的概念要加一些通俗易懂的例子，目标受众是普通小白，PPT 内容和解说稿都用简体中文
