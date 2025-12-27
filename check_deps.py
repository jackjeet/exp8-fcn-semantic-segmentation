# 批量验证语义分割核心依赖是否安装
import importlib

# 要验证的库列表
check_libs = [
    "numpy", "pandas", "matplotlib",
    "cv2", "PIL", "torch", "tensorflow"
]

# 批量验证逻辑
for lib in check_libs:
    try:
        # 特殊处理cv2和PIL（导入名和库名不一致）
        if lib == "cv2":
            mod = importlib.import_module("cv2")
            ver = mod.__version__
        elif lib == "PIL":
            mod = importlib.import_module("PIL")
            ver = mod.__version__
        else:
            mod = importlib.import_module(lib)
            ver = mod.__version__ if hasattr(mod, "__version__") else "未知版本"
        print(f"✅ {lib} 已安装，版本：{ver}")
    except ImportError:
        print(f"❌ {lib} 未安装")