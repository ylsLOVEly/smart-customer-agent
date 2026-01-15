import httpx
import json
from datetime import datetime
from config.settings import APIFOX_API_URL

class ApifoxTool:
    async def create_doc(self, case_data: dict) -> str:
        """创建Apifox文档记录"""
        try:
            # 构建文档数据
            doc_data = {
                "title": f"[故障记录] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "content": self._generate_doc_content(case_data),
                "tags": ["故障", "监控", "API异常"],
                "category": "错误日志"
            }
            
            # 导入配置
            from config.settings import APIFOX_API_URL, APIFOX_API_TOKEN, APIFOX_ENABLE_REAL
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                if APIFOX_ENABLE_REAL and APIFOX_API_TOKEN and "your-apifox-token" not in APIFOX_API_TOKEN:
                    # 真实环境：发送实际请求
                    headers = {
                        "Content-Type": "application/json",
                        "X-Apifox-Api-Token": APIFOX_API_TOKEN
                    }
                    
                    try:
                        response = await client.post(APIFOX_API_URL, json=doc_data, headers=headers)
                        if response.status_code == 200:
                            result = response.json()
                            doc_id = result.get("data", {}).get("id", f"DOC_{datetime.now().strftime('%Y%m%d')}_{case_data['case_id']}")
                            print(f"[Apifox] 创建文档成功: {doc_data['title']}, ID: {doc_id}")
                            return doc_id
                        else:
                            print(f"[Apifox] 创建文档失败: {response.status_code}, {response.text[:100]}")
                            # 降级到模拟模式
                            print(f"[Apifox] 降级到模拟模式...")
                            return self._generate_simulated_doc_id(case_data)
                    except Exception as e:
                        print(f"[Apifox] 创建文档异常: {e}")
                        # 降级到模拟模式
                        return self._generate_simulated_doc_id(case_data)
                else:
                    # 模拟环境
                    print(f"[Apifox] 模拟创建文档: {doc_data['title']}")
                    return self._generate_simulated_doc_id(case_data)
                
        except Exception as e:
            print(f"[Apifox] 创建文档失败: {e}")
            return f"ERROR_{case_data['case_id']}"
    
    def _generate_simulated_doc_id(self, case_data: dict) -> str:
        """生成模拟的文档ID"""
        from datetime import datetime
        return f"DOC_{datetime.now().strftime('%Y%m%d')}_{case_data['case_id']}"
    
    def _generate_doc_content(self, case_data: dict) -> str:
        """生成文档内容"""
        monitor_log = case_data.get("monitor_log", [])

        content = []
        content.append(f"# 故障记录 - {case_data['case_id']}")
        content.append(f"## 基本信息")
        content.append(f"- **故障时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"- **API状态**: {case_data.get('api_status', 'Unknown')}")
        content.append(f"- **响应时间**: {case_data.get('api_response_time', 'N/A')}")
        content.append(f"- **用户查询**: {case_data.get('user_query', '')}")

        if monitor_log:
            content.append(f"\n## 监控日志")
            for log in monitor_log:
                content.append(f"- **时间**: {log.get('timestamp', 'N/A')}")
                content.append(f"  **状态**: {log.get('status', 'N/A')}")
                content.append(f"  **信息**: {log.get('msg', 'N/A')}")

        content.append(f"\n## 处理状态")
        content.append(f"- **记录时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"- **状态**: 已记录到知识库")

        return "\n".join(content)
    
    async def create_error_doc(self, case_id: str, case_data: dict) -> str:
        """
        创建错误文档 - 适配增强版Agent的接口调用
        
        增强版Agent调用：apifox_tool.create_error_doc(case_id, case_data)
        原始方法：create_doc(case_data: dict) -> str
        
        这个方法作为适配器，将增强版Agent的调用转换为原方法调用
        """
        try:
            # 将参数转换为原始方法需要的格式
            case_data_with_id = case_data.copy()
            case_data_with_id['case_id'] = case_id
            
            # 直接调用异步方法
            return await self.create_doc(case_data_with_id)
        except Exception as e:
            print(f"[Apifox] create_error_doc适配器调用失败: {e}")
            return f"ERROR_{case_id}"
