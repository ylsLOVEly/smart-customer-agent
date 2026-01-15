#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能客服监控Agent - 主程序入口
适配比赛提交要求的目录结构
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from agents.customer_agent import CustomerServiceAgent
from agents.enhanced_customer_agent import EnhancedCustomerServiceAgent

async def main():
    """主程序入口 - 处理标准输入并生成输出"""
    print("=" * 60)
    print("智能客服监控Agent - 比赛提交版本（增强版）")
    print("=" * 60)
    
    # 使用增强版Agent
    agent = EnhancedCustomerServiceAgent()
    
    # 输入输出文件路径
    project_root = current_dir.parent
    input_file = project_root / "data" / "inputs.json"
    output_file = project_root / "data" / "outputs" / "results.json"
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    try:
        # 读取输入数据
        with open(input_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        print(f"📊 读取到 {len(test_cases)} 个测试案例")
        
        results = []
        for i, case_data in enumerate(test_cases, 1):
            print(f"\n🔄 处理案例 {i}/{len(test_cases)}: {case_data['case_id']}")
            print(f"   用户问题: {case_data['user_query'][:50]}...")
            
            # 处理案例
            result = await agent.process_case(case_data)
            results.append(result)
            
            # 显示处理结果
            print(f"   Agent回复: {result['reply'][:100]}...")
            if result.get('action_triggered'):
                print(f"   触发动作: {len(result['action_triggered'])}个")
            else:
                print(f"   触发动作: 无")
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 处理完成！结果已保存到: {output_file}")
        print(f"📈 总计处理: {len(results)}个案例")
        
        # 统计结果
        alert_count = sum(1 for r in results if r.get('action_triggered'))
        print(f"🚨 触发告警: {alert_count}个案例")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
