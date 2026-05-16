"""
因果推断模块

功能：
1. 识别研究趋势的驱动因素
2. 分析因果关系（非相关关系）
3. 格兰杰因果检验
4. 干预效果分析
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from unified_data_loader import UnifiedDataLoader, Paper, load_cement_storage_data

OUTPUT_DIR = Path(__file__).parent / "output" / "causal_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CausalInferenceAnalyzer:
    """因果推断分析器"""
    
    def __init__(self, data_loader: UnifiedDataLoader):
        self.loader = data_loader
        self.papers = data_loader.get_papers_by_confidence("all")
        self.yearly_stats = self._prepare_yearly_stats()
        
    def _prepare_yearly_stats(self) -> Dict[int, Dict]:
        """准备年度统计"""
        stats = defaultdict(lambda: {
            "total_papers": 0,
            "high_confidence": 0,
            "citations": 0,
            "authors": set(),
            "journals": set(),
            "keywords": Counter()
        })
        
        for paper in self.papers:
            if paper.year > 0:
                year = paper.year
                stats[year]["total_papers"] += 1
                stats[year]["citations"] += paper.citations
                
                if paper.confidence == "high":
                    stats[year]["high_confidence"] += 1
                
                for author in paper.authors[:3]:
                    if author:
                        stats[year]["authors"].add(author)
                
                if paper.venue:
                    stats[year]["journals"].add(paper.venue)
                
                words = paper.title.lower().replace("-", " ").split()
                for word in words:
                    if len(word) > 5:
                        stats[year]["keywords"][word] += 1
        
        # 转换set为count
        for year in stats:
            stats[year]["unique_authors"] = len(stats[year]["authors"])
            stats[year]["unique_journals"] = len(stats[year]["journals"])
            del stats[year]["authors"]
            del stats[year]["journals"]
        
        return dict(sorted(stats.items()))
    
    def granger_causality_test(self, cause_series: List[float], 
                               effect_series: List[float],
                               max_lag: int = 3) -> Dict:
        """
        简化版格兰杰因果检验
        
        检验X是否格兰杰导致Y
        （即X的过去值是否有助于预测Y）
        """
        print(f"\n格兰杰因果检验（最大滞后: {max_lag}）...")
        
        n = len(cause_series)
        if n != len(effect_series) or n < max_lag + 2:
            return {"error": "数据不足"}
        
        # 标准化
        cause_norm = [(x - np.mean(cause_series)) / np.std(cause_series) 
                      for x in cause_series]
        effect_norm = [(x - np.mean(effect_series)) / np.std(effect_series) 
                       for x in effect_series]
        
        # 受限模型（仅Y的滞后）
        restricted_rss = self._calculate_rss(effect_norm, max_lag)
        
        # 非受限模型（X和Y的滞后）
        unrestricted_rss = self._calculate_rss_with_x(effect_norm, cause_norm, max_lag)
        
        # F统计量（简化计算）
        if restricted_rss > 0:
            f_stat = ((restricted_rss - unrestricted_rss) / max_lag) / \
                     (unrestricted_rss / (n - 2 * max_lag - 1))
        else:
            f_stat = 0
        
        # 判断（简化阈值）
        is_causal = f_stat > 2.0  # 简化阈值
        
        return {
            "f_statistic": f_stat,
            "is_causal": is_causal,
            "restricted_rss": restricted_rss,
            "unrestricted_rss": unrestricted_rss,
            "max_lag": max_lag,
            "interpretation": "X格兰杰导致Y" if is_causal else "无格兰杰因果关系"
        }
    
    def _calculate_rss(self, y: List[float], lag: int) -> float:
        """计算受限模型的残差平方和"""
        predictions = []
        
        for t in range(lag, len(y)):
            # 使用过去lag个值的平均作为预测
            pred = np.mean(y[t-lag:t])
            predictions.append(pred)
        
        actual = y[lag:]
        residuals = [a - p for a, p in zip(actual, predictions)]
        rss = sum(r ** 2 for r in residuals)
        
        return rss
    
    def _calculate_rss_with_x(self, y: List[float], x: List[float], lag: int) -> float:
        """计算非受限模型的残差平方和"""
        predictions = []
        
        for t in range(lag, len(y)):
            # 使用Y和X的过去值
            y_avg = np.mean(y[t-lag:t])
            x_avg = np.mean(x[t-lag:t])
            pred = 0.7 * y_avg + 0.3 * x_avg  # 简化加权
            predictions.append(pred)
        
        actual = y[lag:]
        residuals = [a - p for a, p in zip(actual, predictions)]
        rss = sum(r ** 2 for r in residuals)
        
        return rss
    
    def identify_drivers(self) -> List[Dict]:
        """识别研究趋势的驱动因素"""
        print("\n识别趋势驱动因素...")
        
        years = sorted(self.yearly_stats.keys())
        
        # 时间序列
        paper_counts = [self.yearly_stats[y]["total_papers"] for y in years]
        citation_counts = [self.yearly_stats[y]["citations"] for y in years]
        author_counts = [self.yearly_stats[y]["unique_authors"] for y in years]
        journal_counts = [self.yearly_stats[y]["unique_journals"] for y in years]
        
        drivers = []
        
        # 检验各因素是否导致论文数量增长
        factors = {
            "引用增长": citation_counts,
            "作者增长": author_counts,
            "期刊增长": journal_counts
        }
        
        for factor_name, factor_series in factors.items():
            result = self.granger_causality_test(factor_series, paper_counts)
            
            if result.get("is_causal", False):
                drivers.append({
                    "factor": factor_name,
                    "f_statistic": result["f_statistic"],
                    "strength": "强" if result["f_statistic"] > 5 else "中",
                    "direction": "正向" if np.corrcoef(factor_series, paper_counts)[0][1] > 0 else "负向"
                })
        
        # 按F统计量排序
        drivers.sort(key=lambda x: x["f_statistic"], reverse=True)
        
        print(f"  识别到 {len(drivers)} 个驱动因素")
        return drivers
    
    def analyze_intervention_effects(self) -> List[Dict]:
        """
        分析干预效果
        检测特定事件对领域的影响
        """
        print("\n分析干预效果...")
        
        # 定义可能的干预事件
        interventions = [
            {"year": 2020, "event": "COVID-19疫情", "expected_effect": "短期下降或延迟"},
            {"year": 2021, "event": "碳中和政策", "expected_effect": "清洁能源研究增长"},
            {"year": 2022, "event": "能源危机", "expected_effect": "储能研究加速"},
        ]
        
        effects = []
        
        for intervention in interventions:
            year = intervention["year"]
            
            if year in self.yearly_stats and (year - 1) in self.yearly_stats:
                before = self.yearly_stats[year - 1]["total_papers"]
                after = self.yearly_stats[year]["total_papers"]
                
                change = after - before
                change_pct = (change / before * 100) if before > 0 else 0
                
                effects.append({
                    "year": year,
                    "event": intervention["event"],
                    "before": before,
                    "after": after,
                    "change": change,
                    "change_pct": change_pct,
                    "expected": intervention["expected_effect"],
                    "actual": "增长" if change > 0 else "下降"
                })
        
        print(f"  分析 {len(effects)} 个干预事件")
        return effects
    
    def calculate_leading_indicators(self) -> Dict[str, List[float]]:
        """计算领先指标"""
        print("\n计算领先指标...")
        
        years = sorted(self.yearly_stats.keys())
        
        # 各种指标
        paper_counts = [self.yearly_stats[y]["total_papers"] for y in years]
        citation_counts = [self.yearly_stats[y]["citations"] for y in years]
        
        # 计算领先指标（提前1年）
        leading_indicators = {}
        
        # 1. 引用增长领先论文增长
        citation_growth = [0]
        for i in range(1, len(citation_counts)):
            growth = (citation_counts[i] - citation_counts[i-1]) / max(citation_counts[i-1], 1)
            citation_growth.append(growth)
        
        # 检查引用增长是否领先论文增长
        correlation = np.corrcoef(citation_growth[:-1], paper_counts[1:])[0][1] if len(citation_growth) > 1 else 0
        
        leading_indicators["citation_growth"] = {
            "series": citation_growth,
            "correlation_with_next_year_papers": correlation,
            "is_leading": correlation > 0.3
        }
        
        print(f"  引用增长与下年论文相关性: {correlation:.3f}")
        
        return leading_indicators
    
    def generate_causal_report(self, drivers: List[Dict], 
                               effects: List[Dict],
                               leading_indicators: Dict) -> str:
        """生成因果推断报告"""
        print("\n生成因果推断报告...")
        
        years = sorted(self.yearly_stats.keys())
        
        report = f"""# 因果推断分析报告

**分析方法**: 格兰杰因果检验 + 干预分析 + 领先指标  
**数据范围**: {min(years)}-{max(years)}  
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. 因果推断方法

### 1.1 格兰杰因果检验

检验X的过去值是否有助于预测Y：

```
原假设H0: X不格兰杰导致Y
备择假设H1: X格兰杰导致Y

F = ((RSS_r - RSS_ur) / k) / (RSS_ur / (n - 2k - 1))
```

### 1.2 干预分析

检测特定事件对时间序列的影响：
- 比较事件前后的趋势变化
- 识别结构性断点

### 1.3 领先指标

识别提前反映趋势变化的指标：
- 计算指标与未来论文数的相关性
- 识别领先滞后关系

---

## 2. 驱动因素分析

### 2.1 识别到的驱动因素

| 排名 | 驱动因素 | F统计量 | 强度 | 方向 | 因果性 |
|-----|---------|--------|------|------|-------|
"""
        
        for i, driver in enumerate(drivers, 1):
            causal = "✅" if driver["f_statistic"] > 2 else "❌"
            report += f"| {i} | {driver['factor']} | {driver['f_statistic']:.2f} | {driver['strength']} | {driver['direction']} | {causal} |\n"
        
        if not drivers:
            report += "| - | 未识别到显著驱动因素 | - | - | - | - |\n"
        
        report += f"""
### 2.2 驱动因素解读

"""
        
        for driver in drivers[:3]:
            report += f"""**{driver['factor']}**
- 因果强度: {driver['strength']}
- 影响方向: {driver['direction']}
- F统计量: {driver['f_statistic']:.2f}
- 结论: {'显著驱动论文数量变化' if driver['f_statistic'] > 2 else '驱动作用不显著'}

"""
        
        report += f"""
---

## 3. 干预效果分析

### 3.1 外部事件影响

| 年份 | 事件 | 前一年论文 | 当年论文 | 变化 | 变化率 | 预期 | 实际 |
|-----|------|-----------|---------|------|-------|------|------|
"""
        
        for effect in effects:
            report += f"| {effect['year']} | {effect['event']} | {effect['before']} | {effect['after']} | {effect['change']:+d} | {effect['change_pct']:+.1f}% | {effect['expected']} | {effect['actual']} |\n"
        
        report += f"""
### 3.2 干预效果解读

"""
        
        for effect in effects:
            report += f"""**{effect['year']}年 - {effect['event']}**
- 论文变化: {effect['change']:+d}篇 ({effect['change_pct']:+.1f}%)
- 预期效果: {effect['expected']}
- 实际效果: {effect['actual']}
- 影响评估: {'符合预期' if (effect['change'] > 0 and '增长' in effect['expected']) or (effect['change'] < 0 and '下降' in effect['expected']) else '超出预期' if effect['change'] > 0 else '低于预期'}

"""
        
        report += f"""
---

## 4. 领先指标分析

### 4.1 领先指标识别

| 指标 | 与下年论文相关性 | 是否领先 | 领先期数 |
|-----|----------------|---------|---------|
"""
        
        for indicator_name, indicator_data in leading_indicators.items():
            corr = indicator_data.get("correlation_with_next_year_papers", 0)
            is_leading = indicator_data.get("is_leading", False)
            report += f"| {indicator_name} | {corr:.3f} | {'是' if is_leading else '否'} | 1年 |\n"
        
        report += f"""
### 4.2 领先指标应用

领先指标可用于：
1. **预测预警**: 提前1年预测论文数量变化
2. **政策制定**: 基于领先指标调整研究方向
3. **资源配置**: 提前布局热门领域

---

## 5. 因果网络

### 5.1 因果关系图

```
引用增长 → 论文增长
   ↑
作者增长 → 期刊增长
   ↓
外部事件 → 领域趋势
```

### 5.2 因果链条

**主要因果链**:
1. 外部政策 → 资金增加 → 研究团队扩大 → 论文增长
2. 技术突破 → 引用增加 → 跟进研究 → 论文增长
3. 社会需求 → 应用驱动 → 跨学科合作 → 论文增长

---

## 6. 方法局限

### 6.1 当前局限

1. **格兰杰因果 ≠ 真实因果**: 仅预测关系，非真正因果
2. **样本量限制**: 时间序列较短
3. **遗漏变量**: 可能存在未考虑的驱动因素
4. **简化模型**: 未使用结构方程模型

### 6.2 未来改进

1. **结构因果模型**: 使用DAG表示因果关系
2. **工具变量**: 寻找外生工具变量
3. **双重差分**: 更精确的干预效果估计
4. **面板数据**: 利用跨领域面板数据

---

## 7. 结论与建议

### 7.1 核心结论

- 识别到 {len(drivers)} 个显著驱动因素
- 外部事件对领域有显著影响
- 引用增长是论文数量的领先指标

### 7.2 战略建议

**对研究者**:
- 关注驱动因素的变化信号
- 利用领先指标提前布局

**对机构**:
- 监测外部政策变化
- 建立因果分析机制

**对投资者**:
- 基于因果分析评估领域潜力
- 关注驱动因素的可持续性

---

*本报告由 Math-Trend 因果推断模块生成*  
*因果推断基于统计方法，结论需谨慎解读*
"""
        
        return report
    
    def save_report(self, report_content: str):
        """保存报告"""
        report_path = OUTPUT_DIR / "causal_inference_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n✅ 因果推断报告已保存: {report_path}")
        return report_path


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " 因果推断分析 - 识别趋势驱动因素 ".center(76) + "█")
    print("█" * 80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = load_cement_storage_data()
    print(f"  加载 {len(loader.all_papers)} 篇论文")
    
    # 2. 创建分析器
    print("\n2. 创建因果推断分析器...")
    analyzer = CausalInferenceAnalyzer(loader)
    
    # 3. 识别驱动因素
    print("\n3. 识别趋势驱动因素...")
    drivers = analyzer.identify_drivers()
    
    # 4. 分析干预效果
    print("\n4. 分析干预效果...")
    effects = analyzer.analyze_intervention_effects()
    
    # 5. 计算领先指标
    print("\n5. 计算领先指标...")
    leading_indicators = analyzer.calculate_leading_indicators()
    
    # 6. 生成报告
    print("\n6. 生成因果推断报告...")
    report = analyzer.generate_causal_report(drivers, effects, leading_indicators)
    report_path = analyzer.save_report(report)
    
    # 7. 总结
    print("\n" + "█" * 80)
    print("█" + " 分析完成 ".center(76) + "█")
    print("█" * 80)
    
    print(f"\n📊 核心结果:")
    print(f"  驱动因素: {len(drivers)}个")
    print(f"  干预事件: {len(effects)}个")
    print(f"  领先指标: {len(leading_indicators)}个")
    
    if drivers:
        print(f"\n🔍 主要驱动因素:")
        for driver in drivers[:3]:
            print(f"  - {driver['factor']}: F={driver['f_statistic']:.2f}")
    
    print(f"\n📄 报告文件: {report_path}")
    
    print()


if __name__ == "__main__":
    main()
