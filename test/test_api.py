# test_api.py（临时测试文件）
import asyncio
from agent.models.deepseek_client import DeepSeekClient

async def test():
    client = DeepSeekClient()
    response = await client.call_model(
        model="deepseek/deepseek-v3.2",
        messages=[{"role": "user", "content": "你好"}]
    )
    print("API响应:", response)

asyncio.run(test())
