"""
Math-Trend 模块接口定义（插件协议）

所有分析模块必须实现此接口
"""

from abc import ABC, abstractmethod
from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig


class BaseAnalyzer(ABC):
    """
    分析模块基类
    
    每个分析模块只需实现三个方法：
    - name(): 模块名称
    - analyze(): 执行分析
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    @abstractmethod
    def name(self) -> str:
        """返回模块名称"""
        pass
    
    @abstractmethod
    def analyze(self, papers: list[Paper]) -> AnalysisResult:
        """
        执行分析
        
        Args:
            papers: 已预处理的论文列表
        
        Returns:
            分析结果（统一格式）
        """
        pass
