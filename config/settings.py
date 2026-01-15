# settings.py
import os
import logging
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 🚨 安全修复：API密钥从环境变量读取，不在代码中硬编码
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://router.shengsuanyun.com/api/v1")

# 配置验证函数
def validate_required_config():
    """验证必需的配置项是否存在"""
    required_configs = {
        'DEEPSEEK_API_KEY': DEEPSEEK_API_KEY,
        'DEEPSEEK_BASE_URL': DEEPSEEK_BASE_URL,
    }
    
    missing_configs = []
    for key, value in required_configs.items():
        if not value or value.strip() == '':
            missing_configs.append(key)
    
    if missing_configs:
        raise ValueError(f"缺少必需的配置项: {', '.join(missing_configs)}")
    
    # 验证API密钥格式
    if not DEEPSEEK_API_KEY or len(DEEPSEEK_API_KEY) < 10:
        raise ValueError("DEEPSEEK_API_KEY 格式无效")
    
    logging.info("✅ 配置验证通过")

# 工具配置
# Uptime Kuma配置（模拟或真实）
UPTIME_KUMA_URL = os.getenv("UPTIME_KUMA_URL", "http://localhost:3001")  # Uptime Kuma地址
UPTIME_KUMA_API_KEY = os.getenv("UPTIME_KUMA_API_KEY", "")  # API密钥（如果需要）

# 飞书Webhook配置
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-key")
FEISHU_ENABLE_REAL = os.getenv("FEISHU_ENABLE_REAL", "false").lower() == "true"  # 是否启用真实发送

# Apifox配置
APIFOX_API_URL = os.getenv("APIFOX_API_URL", "https://api.apifox.com/v1/projects/your-project-id/apis")
APIFOX_API_TOKEN = os.getenv("APIFOX_API_TOKEN", "your-apifox-token")
APIFOX_ENABLE_REAL = os.getenv("APIFOX_ENABLE_REAL", "false").lower() == "true"  # 是否启用真实发送

# 运行模式
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"  # 是否使用模拟模式

# 路径配置
INPUT_FILE = "data/inputs.json"
OUTPUT_FILE = "data/outputs/results.json"
KNOWLEDGE_BASE_PATH = "agent/knowledge_base/"

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "logs/agent.log")

def setup_logging():
    """配置统一的日志系统"""
    import os
    from pathlib import Path
    
    # 确保日志目录存在
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=log_format,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info(f"日志系统初始化完成，级别: {LOG_LEVEL}，文件: {LOG_FILE}")

# 初始化配置和日志
def initialize_system():
    """初始化系统配置和日志"""
    setup_logging()
    validate_required_config()
    logging.info("智能客服监控Agent系统初始化完成")
