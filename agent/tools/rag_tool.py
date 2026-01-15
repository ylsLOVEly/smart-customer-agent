import json
import os
from typing import Optional
from pathlib import Path

class RAGTool:
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> list:
        """加载JSON格式的知识库"""
        try:
            # 修复路径问题 - 支持多个可能的知识库位置
            possible_paths = [
                Path("agent/knowledge_base/platform_knowledge.json"),  # agent目录结构
                Path("knowledge_base/platform_knowledge.json"),        # 传统结构
                Path("data/knowledge_base/platform_knowledge.json")     # data目录结构
            ]
            
            knowledge_file = None
            for path in possible_paths:
                if path.exists():
                    knowledge_file = path
                    break
            
            if knowledge_file:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    knowledge_data = data.get("platform_knowledge", [])
                    print(f"[RAG] 成功加载知识库: {knowledge_file} ({len(knowledge_data)}条记录)")
                    return knowledge_data
            else:
                print(f"[RAG] 知识库文件未找到，使用默认知识库")
                return self._get_default_knowledge()
        except Exception as e:
            print(f"[RAG] 加载知识库失败: {e}")
            return self._get_default_knowledge()
    
    def _get_default_knowledge(self) -> list:
        """获取默认知识库"""
        return [
            {
                "category": "计费模式",
                "keywords": ["计费", "费用", "价格", "订阅", "包月", "付费"],
                "content": "平台提供按量付费和包月订阅两种模式...详细内容请参考官方文档。"
            },
            {
                "category": "系统稳定性",
                "keywords": ["稳定", "可用性", "宕机", "故障", "监控", "状态"],
                "content": "系统采用多可用区部署，保证99.9%可用性...故障时会自动告警和处理。"
            }
        ]
    
    def search(self, query: str) -> str:
        """搜索知识库，返回最相关的内容"""
        query_lower = query.lower()
        matched_items = []
        
        # 第一轮：精确匹配关键词
        for item in self.knowledge_base:
            keywords = item.get("keywords", [])
            for keyword in keywords:
                if keyword in query_lower:
                    matched_items.append(item)
                    break
        
        # 第二轮：类别匹配
        if not matched_items:
            for item in self.knowledge_base:
                category = item.get("category", "").lower()
                if category in query_lower:
                    matched_items.append(item)
                    break
        
        # 第三轮：部分关键词匹配
        if not matched_items:
            for item in self.knowledge_base:
                content = item.get("content", "").lower()
                keywords = item.get("keywords", [])
                # 检查内容中是否包含查询的关键词
                for word in query_lower.split():
                    if len(word) > 2 and (word in content or any(word in kw for kw in keywords)):
                        matched_items.append(item)
                        break
        
        if matched_items:
            # 返回匹配度最高的内容
            best_match = matched_items[0]
            # 如果查询中包含疑问词，提供更友好的回答
            if any(q_word in query_lower for q_word in ["怎么", "如何", "怎样", "为什么"]):
                return f"关于{best_match['category']}：{best_match['content']}"
            else:
                return best_match['content']
        
        return "很抱歉，在知识库中未找到相关信息。您可以尝试以下方式获取帮助：\n1. 查看官方文档\n2. 联系在线客服\n3. 在社区论坛提问"
