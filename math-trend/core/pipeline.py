"""
Math-Trend Pipeline引擎

端到端工作流编排：
  数据采集/缓存 → 质量分类 → 预处理 → 分析 → 报告
"""

import importlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from core.config import AnalysisConfig
from core.data_model import Paper, AnalysisResult
from core.interfaces import BaseAnalyzer
from data.collector import DataCollector
from data.cache import DataCache
from data.quality import QualityClassifier


class PipelineEngine:
    """Pipeline引擎"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results: Dict[str, AnalysisResult] = {}
        self.papers: List[Paper] = []
    
    def run(self, force_refresh: bool = False) -> Dict[str, AnalysisResult]:
        """运行完整Pipeline"""
        print("\n" + "=" * 70)
        print(f"【Math-Trend Pipeline - {self.config.domain_name}】")
        print("=" * 70)
        
        # Step 1: 数据加载（缓存优先）
        self.papers = self._load_data(force_refresh)
        print(f"\n✅ 数据就绪: {len(self.papers)}篇")
        
        # Step 2: 质量分类
        self.papers = self._classify_quality(self.papers)
        high = sum(1 for p in self.papers if p.confidence == 'high')
        medium = sum(1 for p in self.papers if p.confidence == 'medium')
        low = sum(1 for p in self.papers if p.confidence == 'low')
        print(f"✅ 质量分类: 高{high}/中{medium}/低{low}")
        
        # Step 3: 预处理
        self.papers = self._preprocess(self.papers)
        print(f"✅ 预处理完成")
        
        # Step 4: 运行分析模块
        print(f"\n{'='*70}")
        print(f"【运行分析模块】")
        print(f"{'='*70}")
        
        modules = self._load_modules()
        print(f"已注册 {len(modules)} 个模块")
        
        for module in modules:
            self._run_module(module, self.papers)
        
        return self.results
    
    def _load_data(self, force_refresh: bool = False) -> List[Paper]:
        """加载数据（缓存 → 已分类数据 → 实时采集）"""
        print("\n[1/4] 数据加载...")
        
        cache = DataCache()
        
        # 1. 尝试缓存
        if not force_refresh and cache.is_fresh(self.config):
            papers = cache.load_papers(self.config)
            if papers:
                return papers
        
        # 2. 尝试已分类数据
        classified_path = Path(__file__).parent.parent / "output" / "analysis_608" / "classified_papers.json"
        if classified_path.exists():
            print(f"  使用已分类数据: {classified_path}")
            papers = self._load_classified_papers(classified_path)
            if papers:
                cache.save_papers(papers, self.config)
                return papers
        
        # 3. 实时采集
        print("  已有数据不可用，从数据源采集...")
        collector = DataCollector(self.config)
        papers = collector.collect()
        
        if not papers:
            print("  ⚠️ 未采集到数据")
        
        return papers
    
    def _load_classified_papers(self, path: Path) -> List[Paper]:
        """加载已分类论文"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        papers = []
        for level in ["high_confidence", "medium_confidence", "low_confidence"]:
            for p in data["papers"].get(level, []):
                papers.append(Paper.from_dict(p))
        
        return papers
    
    def _classify_quality(self, papers: List[Paper]) -> List[Paper]:
        """质量分类"""
        print("\n[2/4] 质量分类...")
        
        # 如果已有分类，跳过
        if all(p.confidence in ('high', 'medium', 'low') and p.confidence_reason != "待分类" for p in papers[:5]):
            print("  已有分类，跳过")
            return papers
        
        classifier = QualityClassifier(self.config)
        return classifier.classify(papers)
    
    def _preprocess(self, papers: List[Paper]) -> List[Paper]:
        """预处理（只做一次）"""
        print("\n[3/4] 数据预处理...")
        
        for paper in papers:
            if not paper.keywords:
                paper.keywords = self._extract_keywords(paper)
        
        return papers
    
    def _extract_keywords(self, paper: Paper) -> List[str]:
        """从标题和摘要提取关键词"""
        text = f"{paper.title} {paper.abstract}".lower()
        important_terms = [
            'supercapacitor', 'battery', 'energy storage', 'conductive',
            'carbon nanotube', 'graphene', 'cement', 'concrete',
            'multifunctional', 'structural', 'electrode', 'electrolyte',
            'PEDOT', 'CNT', 'activated carbon', 'fly ash',
            'compressive strength', 'capacitance', 'energy density',
            'power density', 'cyclability', 'self-sensing', 'thermal'
        ]
        return [term for term in important_terms if term in text]
    
    def _load_modules(self) -> List[BaseAnalyzer]:
        """动态加载分析模块"""
        modules = []
        
        for module_path in self.config.enabled_modules:
            try:
                full_path = f"analysis.{module_path}"
                mod = importlib.import_module(full_path)
                if hasattr(mod, 'get_analyzer'):
                    analyzer = mod.get_analyzer(self.config)
                    modules.append(analyzer)
                else:
                    print(f"  ⚠️ {module_path}: 缺少 get_analyzer() 函数")
            except ImportError as e:
                print(f"  ⚠️ {module_path}: 导入失败 ({e})")
            except Exception as e:
                print(f"  ⚠️ {module_path}: 加载失败 ({e})")
        
        return modules
    
    def _run_module(self, analyzer: BaseAnalyzer, papers: List[Paper]):
        """运行单个分析模块"""
        print(f"  [{analyzer.name()}] 运行中...", end=" ")
        
        try:
            result = analyzer.analyze(papers)
            self.results[analyzer.name()] = result
            
            if result.success:
                print(f"✅ ({len(result.data)}项结果)")
            else:
                print(f"⚠️ ({result.error})")
        except Exception as e:
            print(f"❌ (异常: {e})")
            self.results[analyzer.name()] = AnalysisResult(
                module_name=analyzer.name(),
                timestamp=datetime.now(),
                success=False,
                data={},
                error=str(e)
            )
    
    def save_results(self, output_dir: Path | None = None):
        """保存分析结果"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / self.config.output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_dict = {}
        for name, result in self.results.items():
            results_dict[name] = {
                "success": result.success,
                "data": result.data,
                "metadata": result.metadata,
                "error": result.error,
            }
        
        output_path = output_dir / "pipeline_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存: {output_path}")
        return output_path
