class MonitorTool:
    def check_status(self, api_status: str, monitor_log: list) -> dict:
        """检查是否需要触发告警"""
        result = {
            "need_alert": False,
            "alert_reason": None,
            "latest_error": None
        }
        
        if api_status != "200 OK":
            result["need_alert"] = True
            result["alert_reason"] = f"API状态异常: {api_status}"
        
        if monitor_log:
            # 检查最近的错误日志
            for log in reversed(monitor_log):
                if log.get("status") == "Error":
                    result["need_alert"] = True
                    result["latest_error"] = log
                    break
        
        return result