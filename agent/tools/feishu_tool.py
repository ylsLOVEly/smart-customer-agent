import httpx
import json
from datetime import datetime
from config.settings import FEISHU_WEBHOOK_URL

class FeishuTool:
    async def send_alert(self, case_data: dict) -> str:
        """构造并发送飞书卡片消息"""
        try:
            # 获取错误信息
            monitor_log = case_data.get("monitor_log", [])
            latest_error = None
            if monitor_log:
                for log in reversed(monitor_log):
                    if log.get("status") == "Error":
                        latest_error = log
                        break
            
            # 构建卡片消息
            card = self._build_feishu_card(case_data, latest_error)
            
            # 发送请求
            from config.settings import FEISHU_WEBHOOK_URL, FEISHU_ENABLE_REAL
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                if FEISHU_ENABLE_REAL and FEISHU_WEBHOOK_URL and "your-webhook-key" not in FEISHU_WEBHOOK_URL:
                    # 真实环境：发送实际请求
                    try:
                        response = await client.post(FEISHU_WEBHOOK_URL, json=card)
                        if response.status_code == 200:
                            print(f"[飞书] 发送告警成功: {case_data['case_id']}")
                            return f"Sent success (Real: {response.status_code})"
                        else:
                            print(f"[飞书] 发送告警失败: {response.status_code}")
                            return f"Error: HTTP {response.status_code}"
                    except Exception as e:
                        print(f"[飞书] 发送请求异常: {e}")
                        return f"Error: {str(e)}"
                else:
                    # 模拟环境：仅打印日志
                    print(f"[飞书] 模拟发送告警: {case_data['case_id']}")
                    print(f"   目标URL: {FEISHU_WEBHOOK_URL}")
                    print(f"   卡片内容: {json.dumps(card, ensure_ascii=False, indent=2)[:200]}...")
                    return "Sent success (Simulation)"
                
        except Exception as e:
            print(f"[飞书] 发送告警失败: {e}")
            return f"Error: {str(e)}"
    
    def _build_feishu_card(self, case_data: dict, latest_error: dict = None) -> dict:
        """构建飞书卡片消息"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 基础卡片结构
        card = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "🚨 系统故障告警"
                    },
                    "template": "red"
                },
                "elements": []
            }
        }
        
        # 添加时间信息
        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**告警时间**: {current_time}\n"
                              f"**案例ID**: {case_data['case_id']}"
                }
            },
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**API状态**: {case_data.get('api_status', 'Unknown')}\n"
                              f"**响应时间**: {case_data.get('api_response_time', 'N/A')}"
                }
            }
        ]
        
        # 添加错误信息
        if latest_error:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**最近错误**:\n"
                              f"时间: {latest_error.get('timestamp', 'N/A')}\n"
                              f"状态: {latest_error.get('status', 'N/A')}\n"
                              f"信息: {latest_error.get('msg', 'N/A')}"
                }
            })
        
        # 添加用户查询
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**用户查询**: {case_data.get('user_query', 'N/A')}"
            }
        })
        
        # 添加处理状态
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": "已触发自动化处理流程，相关文档正在生成中..."
                }
            ]
        })
        
        card["card"]["elements"] = elements
        return card
