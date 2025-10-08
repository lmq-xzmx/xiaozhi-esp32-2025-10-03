import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)

def auto_import_modules(package_name):
    """
    自动导入指定包下的所有模块
    
    Args:
        package_name: 包名，如 'plugins_func.functions'
    """
    try:
        # 导入包
        package = importlib.import_module(package_name)
        
        # 获取包的路径
        if hasattr(package, '__path__'):
            package_path = package.__path__
        else:
            logger.warning(f"Package {package_name} has no __path__ attribute")
            return
        
        # 遍历包中的所有模块
        for importer, modname, ispkg in pkgutil.iter_modules(package_path):
            if not ispkg:  # 只导入模块，不导入子包
                full_module_name = f"{package_name}.{modname}"
                try:
                    importlib.import_module(full_module_name)
                    logger.debug(f"Successfully imported module: {full_module_name}")
                except Exception as e:
                    logger.error(f"Failed to import module {full_module_name}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Failed to auto import modules from {package_name}: {str(e)}")


def load_all_plugins():
    """加载所有插件"""
    try:
        auto_import_modules("plugins_func.functions")
        logger.info("All plugins loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load plugins: {str(e)}")