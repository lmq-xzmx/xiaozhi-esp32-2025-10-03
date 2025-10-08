import logging

# 使用标准logging而不是loguru
logger = logging.getLogger(__name__)

TAG = "PLUGIN_REGISTER"

class Action:
    def __init__(self, action_type, data=None):
        self.action_type = action_type
        self.data = data or {}

class ActionResult:
    def __init__(self, action, result, response=None):
        self.action = action  # 动作类型
        self.result = result  # 动作产生的结果
        self.response = response  # 直接回复的内容

class ActionResponse:
    def __init__(self, success=True, message="", data=None):
        self.success = success
        self.message = message
        self.data = data or {}


class FunctionItem:
    def __init__(self, name, description, func, type):
        self.name = name
        self.description = description
        self.func = func
        self.type = type


class DeviceTypeRegistry:
    """设备类型注册表，用于管理IOT设备类型及其函数"""

    def __init__(self):
        self.type_functions = {}  # type_signature -> {func_name: FunctionItem}

    def generate_device_type_id(self, descriptor):
        """通过设备能力描述生成类型ID"""
        properties = sorted(descriptor["properties"].keys())
        methods = sorted(descriptor["methods"].keys())
        # 使用属性和方法的组合作为设备类型的唯一标识
        type_signature = (
            f"{descriptor['name']}:{','.join(properties)}:{','.join(methods)}"
        )
        return type_signature

    def get_device_functions(self, type_id):
        """获取设备类型对应的所有函数"""
        return self.type_functions.get(type_id, {})

    def register_device_type(self, type_id, functions):
        """注册设备类型及其函数"""
        if type_id not in self.type_functions:
            self.type_functions[type_id] = functions


# 初始化函数注册字典
all_function_registry = {}


def register_function(name, desc, type=None):
    """注册函数到函数注册字典的装饰器"""

    def decorator(func):
        all_function_registry[name] = FunctionItem(name, desc, func, type)
        # 使用标准logging而不是loguru的bind方法
        logger.debug(f"函数 '{name}' 已加载，可以注册使用")
        return func

    return decorator


def register_device_function(name, desc, type=None):
    """注册设备级别的函数到函数注册字典的装饰器"""

    def decorator(func):
        # 使用标准logging而不是loguru的bind方法
        logger.debug(f"设备函数 '{name}' 已加载")
        return func

    return decorator


# 设备类型注册表实例
device_type_registry = DeviceTypeRegistry()