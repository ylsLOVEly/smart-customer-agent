"""
邮箱告警工具 - 通过SMTP发送邮件告警
支持多种邮件服务商：QQ邮箱、163邮箱、Gmail、企业邮箱等
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

class EmailAlertTool:
    """邮件告警工具"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.smtp_server = self.config.get('smtp_server', 'smtp.qq.com')
        self.smtp_port = self.config.get('smtp_port', 465)
        self.use_ssl = self.config.get('use_ssl', True)
        self.sender_email = self.config.get('sender_email', '')
        self.sender_password = self.config.get('sender_password', '')
        self.receiver_emails = self.config.get('receiver_emails', [])
        
        # 验证配置
        self.enabled = all([
            self.sender_email,
            self.sender_password,
            self.receiver_emails
        ])
        
        if not self.enabled:
            self.logger.warning("邮箱告警工具未启用：缺少发件人邮箱、密码或收件人邮箱配置")
        else:
            self.logger.info(f"邮箱告警工具已初始化，发件人：{self.sender_email}")
    
    def _build_email_content(self, case_data: Dict, latest_error: Dict = None) -> Dict[str, str]:
        """构建邮件内容"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        case_id = case_data.get('case_id', 'UNKNOWN')
        api_status = case_data.get('api_status', 'Unknown')
        api_response_time = case_data.get('api_response_time', 'N/A')
        user_query = case_data.get('user_query', '')[ :100] + "..." if len(case_data.get('user_query', '')) > 100 else case_data.get('user_query', '')
        
        # 构建HTML邮件内容
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>系统故障告警 - {case_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .header {{ background-color: #f44336; color: white; padding: 10px; text-align: center; border-radius: 5px 5px 0 0; }}
        .content {{ padding: 20px; }}
        .section {{ margin-bottom: 20px; }}
        .section-title {{ font-weight: bold; color: #555; margin-bottom: 5px; }}
        .section-content {{ background-color: #f9f9f9; padding: 10px; border-left: 3px solid #4CAF50; }}
        .error {{ color: #d32f2f; font-weight: bold; }}
        .warning {{ color: #f57c00; }}
        .info {{ color: #1976d2; }}
        .footer {{ text-align: center; margin-top: 20px; color: #777; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>🚨 系统故障告警</h2>
        </div>
        <div class="content">
            <div class="section">
                <div class="section-title">基本信息</div>
                <div class="section-content">
                    <p><strong>告警时间：</strong> {current_time}</p>
                    <p><strong>案例ID：</strong> {case_id}</p>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">系统状态</div>
                <div class="section-content">
                    <p><strong>API状态：</strong> <span class="error">{api_status}</span></p>
                    <p><strong>响应时间：</strong> {api_response_time}</p>
                </div>
            </div>
"""
        
        if latest_error:
            html_content += f"""
            <div class="section">
                <div class="section-title">错误详情</div>
                <div class="section-content">
                    <p><strong>错误时间：</strong> {latest_error.get('timestamp', 'N/A')}</p>
                    <p><strong>错误状态：</strong> <span class="error">{latest_error.get('status', 'N/A')}</span></p>
                    <p><strong>错误信息：</strong> {latest_error.get('msg', 'N/A')}</p>
                </div>
            </div>
"""
        
        html_content += f"""
            <div class="section">
                <div class="section-title">用户查询</div>
                <div class="section-content">
                    <p><strong>用户问题：</strong> {user_query}</p>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">处理状态</div>
                <div class="section-content info">
                    <p>⚠️ 已触发自动化处理流程，相关文档正在生成中...</p>
                    <p>请检查系统日志并尽快处理此问题。</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>此邮件由智能客服监控Agent自动发送</p>
            <p>请勿直接回复此邮件</p>
        </div>
    </div>
</body>
</html>
"""
        
        # 纯文本版本（备用）
        text_content = f"""
系统故障告警
=============

告警时间：{current_time}
案例ID：{case_id}

系统状态：
- API状态：{api_status}
- 响应时间：{api_response_time}

"""
        
        if latest_error:
            text_content += f"""
错误详情：
- 错误时间：{latest_error.get('timestamp', 'N/A')}
- 错误状态：{latest_error.get('status', 'N/A')}
- 错误信息：{latest_error.get('msg', 'N/A')}

"""
        
        text_content += f"""
用户查询：{user_query}

处理状态：已触发自动化处理流程，相关文档正在生成中。
请检查系统日志并尽快处理此问题。

---
此邮件由智能客服监控Agent自动发送
请勿直接回复此邮件
"""
        
        return {
            'subject': f'🚨 系统故障告警 - {case_id}',
            'html': html_content,
            'text': text_content
        }
    
    async def send_alert(self, case_data: Dict) -> Optional[str]:
        """发送邮件告警"""
        if not self.enabled:
            self.logger.warning("邮箱告警工具未启用，跳过发送")
            return None
        
        try:
            # 获取错误信息
            monitor_log = case_data.get("monitor_log", [])
            latest_error = None
            if monitor_log:
                for log in reversed(monitor_log):
                    if log.get("status") == "Error":
                        latest_error = log
                        break
            
            # 构建邮件内容
            email_content = self._build_email_content(case_data, latest_error)
            
            # 创建邮件消息
            message = MIMEMultipart("alternative")
            message["Subject"] = email_content['subject']
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.receiver_emails)
            
            # 添加纯文本和HTML版本
            part1 = MIMEText(email_content['text'], "plain")
            part2 = MIMEText(email_content['html'], "html")
            message.attach(part1)
            message.attach(part2)
            
            # 发送邮件
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, self.receiver_emails, message.as_string())
            server.quit()
            
            self.logger.info(f"邮件告警发送成功: {case_data['case_id']}, 收件人: {self.receiver_emails}")
            return f"Email sent to {len(self.receiver_emails)} recipients"
            
        except Exception as e:
            self.logger.error(f"邮件告警发送失败: {e}")
            return f"Email error: {str(e)}"
    
    def test_connection(self) -> Dict[str, Any]:
        """测试邮件服务器连接"""
        if not self.enabled:
            return {'success': False, 'message': '邮箱告警工具未启用'}
        
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            server.login(self.sender_email, self.sender_password)
            server.quit()
            
            return {
                'success': True,
                'message': f'连接成功: {self.smtp_server}:{self.smtp_port}',
                'sender': self.sender_email,
                'receivers': self.receiver_emails
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'连接失败: {str(e)}',
                'sender': self.sender_email,
                'receivers': self.receiver_emails
            }

# 需要在模块级别导入
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append('.')
    
    # 测试配置
    test_config = {
        'smtp_server': 'smtp.qq.com',
        'smtp_port': 465,
        'use_ssl': True,
        'sender_email': 'test@qq.com',
        'sender_password': 'your_password',
        'receiver_emails': ['test@example.com']
    }
    
    tool = EmailAlertTool(test_config)
    
    # 测试连接
    result = tool.test_connection()
    print(f"连接测试: {result}")
    
    # 测试告警发送
    test_case = {
        "case_id": "TEST001",
        "user_query": "刚才系统是不是挂了？",
        "api_status": "500 Internal Server Error",
        "api_response_time": "Timeout",
        "monitor_log": [
            {"timestamp": "10:00:01", "status": "Error", "msg": "Connection Refused"}
        ]
    }
    
    # 注意：默认不会真正发送，除非配置真实的邮箱
    if tool.enabled:
        import asyncio
        result = asyncio.run(tool.send_alert(test_case))
        print(f"告警发送结果: {result}")
    else:
        print("邮箱告警工具未启用")
