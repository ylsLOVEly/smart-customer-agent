# settings.py for enhanced agent
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 🚨 安全修复：API密钥从环境变量读取，不在代码中硬编码
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://router.shengsuanyun.com/api/v1")

# 验证必需的配置项
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ DEEPSEEK_API_KEY 环境变量未设置！请检查.env文件")

print(f"✅ DeepSeek配置加载成功，API Key: {DEEPSEEK_API_KEY[:10]}...{DEEPSEEK_API_KEY[-4:]}")

# 工具配置
# Uptime Kuma配置（模拟或真实）
UPTIME_KUMA_URL = os.getenv("UPTIME_KUMA_URL", "http://localhost:3001")
UPTIME_KUMA_API_KEY = os.getenv("UPTIME_KUMA_API_KEY", "")

# 飞书Webhook配置
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-key")
FEISHU_ENABLE_REAL = os.getenv("FEISHU_ENABLE_REAL", "false").lower() == "true"

# Apifox配置
APIFOX_API_URL = os.getenv("APIFOX_API_URL", "https://api.apifox.com/v1/projects/your-project-id/apis")
APIFOX_API_TOKEN = os.getenv("APIFOX_API_TOKEN", "your-apifox-token")
APIFOX_ENABLE_REAL = os.getenv("APIFOX_ENABLE_REAL", "false").lower() == "true"

# 运行模式
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"

# 路径配置
INPUT_FILE = "data/inputs.json"
OUTPUT_FILE = "data/outputs/results.json"
KNOWLEDGE_BASE_PATH = "knowledge_base/"
