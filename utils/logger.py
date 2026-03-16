
import datetime
import sys

IS_LOGGING = True

def log(message):
    """
    日志函数：自动带上 文件名、行号，兼容 f-string 传入
    :param message: 要输出的日志内容（可传 f-string）
    :return: None
    """
    if IS_LOGGING:
        # 获取调用 log() 函数的栈帧（1 表示上一层栈帧，即调用者的位置）
        frame = sys._getframe(1)
        # 提取文件名（basename 只保留文件名，去掉路径）
        filename = frame.f_code.co_filename.split("/")[-1]  # 兼容 Linux/Mac
        # filename = frame.f_code.co_filename.split("\\")[-1]  # 兼容 Windows
        # 提取行号
        line_no = frame.f_lineno
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 拼接日志内容：时间戳 | 文件名:行号 | 日志信息
        log_msg = f"[{timestamp}] [{filename}:{line_no}] {message}"
        print(log_msg)
        
if __name__ == "__main__":
    content = "nihao"
    log(f"test:{content}")