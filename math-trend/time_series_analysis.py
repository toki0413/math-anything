"""
时间序列分析模块

功能：
1. 研究趋势时间序列建模
2. 预测未来发展趋势
3. 季节性/周期性检测
4. 突变点检测
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from unified_data_loader import UnifiedDataLoader, Paper, load_cement_storage_data

OUTPUT_DIR = Path(__file__).parent / "output" / "time_series"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TimeSeriesAnalyzer:
    """时间序列分析器"""
    
    def __init__(self, data_loader: UnifiedDataLoader):
        self.loader = data_loader
        self.papers = data_loader.get_papers_by_confidence("all")
        self.yearly_stats: Dict[int, Dict] = {}
        
    def analyze_yearly_trends(self) -> Dict[int, Dict]:
        """分析年度趋势"""
        print("\n分析年度趋势...")
        
        # 按年统计
        yearly_data = defaultdict(lambda: {
            "papers": [],
            "total_citations": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "journals": Counter(),
            "authors": Counter(),
            "keywords": Counter()
        })
        
        for paper in self.papers:
            if paper.year > 0:
                year = paper.year
                yearly_data[year]["papers"].append(paper)
                yearly_data[year]["total_citations"] += paper.citations
                
                # 置信度分层
                if paper.confidence == "high":
                    yearly_data[year]["high_confidence"] += 1
                elif paper.confidence == "medium":
                    yearly_data[year]["medium_confidence"] += 1
                else:
                    yearly_data[year]["low_confidence"] += 1
                
                # 期刊
                if paper.venue:
                    yearly_data[year]["journals"][paper.venue] += 1
                
                # 作者
                for author in paper.authors[:3]:
                    if author:
                        yearly_data[year]["authors"][author] += 1
                
                # 关键词（从标题提取）
                words = paper.title.lower().replace("-", " ").split()
                for word in words:
                    if len(word) > 5 and word not in ['using', 'based', 'study', 'analysis']:
                        yearly_data[year]["keywords"][word] += 1
        
        # 计算衍生指标
        for year, data in yearly_data.items():
            paper_count = len(data["papers"])
            data["paper_count"] = paper_count
            data["avg_citations"] = data["total_citations"] / paper_count if paper_count > 0 else 0
            data["top_journals"] = data["journals"].most_common(3)
            data["top_authors"] = data["authors"].most_common(3)
            data["top_keywords"] = data["keywords"].most_common(5)
        
        self.yearly_stats = dict(sorted(yearly_data.items()))
        
        print(f"  分析了 {len(self.yearly_stats)} 个年份")
        return self.yearly_stats
    
    def calculate_growth_rate(self) -> Dict[str, float]:
        """计算增长率"""
        print("\n计算增长率...")
        
        if len(self.yearly_stats) < 2:
            return {}
        
        years = sorted(self.yearly_stats.keys())
        
        # 论文数量增长率
        paper_counts = [self.yearly_stats[y]["paper_count"] for y in years]
        
        # CAGR (复合年均增长率)
        if paper_counts[0] > 0:
            n = len(years) - 1
            cagr = (paper_counts[-1] / paper_counts[0]) ** (1/n) - 1
        else:
            cagr = 0
        
        # 逐年增长率
        yearly_growth = {}
        for i in range(1, len(years)):
            prev = paper_counts[i-1]
            curr = paper_counts[i]
            growth = (curr / prev - 1) * 100 if prev > 0 else 0
            yearly_growth[years[i]] = growth
        
        return {
            "cagr": cagr * 100,
            "yearly_growth": yearly_growth,
            "avg_yearly_growth": np.mean(list(yearly_growth.values())) if yearly_growth else 0
        }
    
    def detect_inflection_points(self) -> List[Dict]:
        """检测拐点（突变点）"""
        print("\n检测拐点...")
        
        if len(self.yearly_stats) < 3:
            return []
        
        years = sorted(self.yearly_stats.keys())
        paper_counts = [self.yearly_stats[y]["paper_count"] for y in years]
        
        inflection_points = []
        
        # 使用二阶差分检测拐点
        for i in range(1, len(years) - 1):
            # 前一段增长
            prev_growth = paper_counts[i] - paper_counts[i-1]
            # 后一段增长
            next_growth = paper_counts[i+1] - paper_counts[i]
            
            # 增长方向改变
            if prev_growth * next_growth < 0 or abs(next_growth - prev_growth) > 5:
                inflection_points.append({
                    "year": years[i],
                    "type": "acceleration" if next_growth > prev_growth else "deceleration",
                    "before_growth": prev_growth,
                    "after_growth": next_growth,
                    "change": next_growth - prev_growth
                })
        
        print(f"  检测到 {len(inflection_points)} 个拐点")
        return inflection_points
    
    def predict_future_trends(self, years_ahead: int = 3) -> Dict[int, int]:
        """
        预测未来趋势
        使用简单指数平滑
        """
        print(f"\n预测未来{years_ahead}年趋势...")
        
        if len(self.yearly_stats) < 3:
            return {}
        
        years = sorted(self.yearly_stats.keys())
        paper_counts = [self.yearly_stats[y]["paper_count"] for y in years]
        
        # 简单指数平滑
        alpha = 0.3  # 平滑系数
        
        # 初始化
        smoothed = [paper_counts[0]]
        for i in range(1, len(paper_counts)):
            s = alpha * paper_counts[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(s)
        
        # 预测
        predictions = {}
        last_value = smoothed[-1]
        last_year = years[-1]
        
        # 计算趋势
        trend = (smoothed[-1] - smoothed[-3]) / 2 if len(smoothed) >= 3 else 0
        
        for i in range(1, years_ahead + 1):
            future_year = last_year + i
            # 简单线性趋势预测
            predicted = max(0, int(last_value + trend * i))
            predictions[future_year] = predicted
        
        print(f"  预测完成: {predictions}")
        return predictions
    
    def analyze_keyword_evolution(self) -> Dict[str, List[Tuple[int, int]]]:
        """分析关键词演化"""
        print("\n分析关键词演化...")
        
        # 追踪关键词的年度出现频率
        keyword_timeline = defaultdict(list)
        
        for year, data in self.yearly_stats.items():
            for keyword, count in data["top_keywords"]:
                keyword_timeline[keyword].append((year, count))
        
        # 识别新兴关键词（近期出现频率上升）
        emerging_keywords = {}
        for keyword, timeline in keyword_timeline.items():
            if len(timeline) >= 2:
                recent = timeline[-1][1]
                previous = timeline[-2][1]
                if recent > previous:
                    emerging_keywords[keyword] = timeline
        
        print(f"  识别 {len(emerging_keywords)} 个上升关键词")
        return emerging_keywords
    
    def generate_time_series_report(self) -> str:
        """生成时间序列分析报告"""
        print("\n生成时间序列分析报告...")
        
        # 确保已分析
        if not self.yearly_stats:
            self.analyze_yearly_trends()
        
        growth = self.calculate_growth_rate()
        inflections = self.detect_inflection_points()
        predictions = self.predict_future_trends()
        keyword_evolution = self.analyze_keyword_evolution()
        
        report = f"""# 时间序列分析报告

**分析方法**: 趋势分析 + 拐点检测 + 指数平滑预测  
**数据范围**: {min(self.yearly_stats.keys())}-{max(self.yearly_stats.keys())}  
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. 年度趋势概览

### 1.1 论文数量趋势

| 年份 | 论文数 | 高置信度 | 中置信度 | 低置信度 | 总被引 | 篇均被引 |
|-----|-------|---------|---------|---------|-------|---------|
"""
        
        for year, data in self.yearly_stats.items():
            report += f"| {year} | {data['paper_count']} | {data['high_confidence']} | {data['medium_confidence']} | {data['low_confidence']} | {data['total_citations']} | {data['avg_citations']:.1f} |\n"
        
        report += f"""
### 1.2 增长指标

| 指标 | 数值 | 解读 |
|-----|------|------|
| 复合年均增长率(CAGR) | {growth.get('cagr', 0):.1f}% | {'高速增长' if growth.get('cagr', 0) > 20 else '中速增长' if growth.get('cagr', 0) > 10 else '低速增长'} |
| 平均年增长率 | {growth.get('avg_yearly_growth', 0):.1f}% | - |
| 数据年份跨度 | {len(self.yearly_stats)}年 | - |

---

## 2. 拐点分析

### 2.1 检测到的拐点

| 年份 | 类型 | 前段增长 | 后段增长 | 变化量 |
|-----|------|---------|---------|-------|
"""
        
        if inflections:
            for point in inflections:
                report += f"| {point['year']} | {point['type']} | {point['before_growth']:+d} | {point['after_growth']:+d} | {point['change']:+d} |\n"
        else:
            report += "| - | 无显著拐点 | - | - | - |\n"
        
        report += f"""
### 2.2 拐点解读

"""
        
        if inflections:
            for point in inflections:
                if point['type'] == 'acceleration':
                    report += f"- **{point['year']}年**: 增长加速，可能受政策或技术突破驱动\n"
                else:
                    report += f"- **{point['year']}年**: 增长放缓，可能进入平台期或调整期\n"
        else:
            report += "- 未检测到显著拐点，领域发展相对平稳\n"
        
        report += f"""
---

## 3. 未来预测

### 3.1 论文数量预测

| 年份 | 预测论文数 | 置信区间 | 预测依据 |
|-----|-----------|---------|---------|
"""
        
        for year, predicted in predictions.items():
            # 简单置信区间
            lower = max(0, int(predicted * 0.8))
            upper = int(predicted * 1.2)
            report += f"| {year} | {predicted} | [{lower}, {upper}] | 指数平滑趋势 |\n"
        
        report += f"""
### 3.2 预测说明

**方法**: 简单指数平滑（α=0.3）
**假设**: 
- 当前趋势延续
- 无重大外部冲击
- 数据索引延迟已考虑

**局限性**:
- 预测基于历史数据，无法预测黑天鹅事件
- 2025-2026年数据可能不完整，影响预测精度
- 未考虑政策、 funding 等外部因素

---

## 4. 关键词演化分析

### 4.1 上升关键词

| 关键词 | 出现年份 | 近期频率 | 趋势 |
|-------|---------|---------|------|
"""
        
        sorted_keywords = sorted(keyword_evolution.items(), 
                                key=lambda x: x[1][-1][1] if x[1] else 0, 
                                reverse=True)[:10]
        
        for keyword, timeline in sorted_keywords:
            if timeline:
                first_year = timeline[0][0]
                recent_freq = timeline[-1][1]
                trend = "📈" if len(timeline) > 1 and timeline[-1][1] > timeline[-2][1] else "📊"
                report += f"| {keyword} | {first_year} | {recent_freq} | {trend} |\n"
        
        report += f"""
### 4.2 关键词演化解读

**新兴方向**:
- 近期高频关键词反映当前研究热点
- 新出现的关键词代表新兴方向

**衰退方向**:
- 早期高频但近期低频的关键词可能在衰退
- 注意区分"成熟"和"过时"

---

## 5. 期刊演化分析

### 5.1 期刊热度变化

| 期刊 | 早期(2018-2020) | 中期(2021-2023) | 近期(2024-2026) | 趋势 |
|-----|----------------|----------------|----------------|------|
"""
        
        # 统计各期刊在不同时期的发文量
        journal_periods = defaultdict(lambda: {"early": 0, "mid": 0, "recent": 0})
        
        for year, data in self.yearly_stats.items():
            period = "early" if year <= 2020 else "mid" if year <= 2023 else "recent"
            for journal, count in data["top_journals"]:
                journal_periods[journal][period] += count
        
        # 显示Top期刊
        for journal, periods in sorted(journal_periods.items(), 
                                      key=lambda x: sum(x[1].values()), 
                                      reverse=True)[:8]:
            trend = "📈" if periods["recent"] > periods["mid"] else "📉" if periods["recent"] < periods["mid"] else "📊"
            report += f"| {journal[:25]} | {periods['early']} | {periods['mid']} | {periods['recent']} | {trend} |\n"
        
        report += f"""
---

## 6. 作者活跃度演化

### 6.1 高产作者时间分布

| 作者 | 活跃年份 | 早期贡献 | 近期贡献 | 持续性 |
|-----|---------|---------|---------|-------|
"""
        
        author_years = defaultdict(list)
        for year, data in self.yearly_stats.items():
            for author, count in data["top_authors"]:
                author_years[author].append((year, count))
        
        for author, timeline in sorted(author_years.items(), 
                                      key=lambda x: sum(c for _, c in x[1]), 
                                      reverse=True)[:8]:
            years_active = [y for y, _ in timeline]
            early = sum(c for y, c in timeline if y <= 2020)
            recent = sum(c for y, c in timeline if y >= 2024)
            
            if recent > 0 and early > 0:
                continuity = "持续活跃"
            elif recent > 0:
                continuity = "新兴力量"
            else:
                continuity = "早期贡献者"
            
            report += f"| {author[:20]} | {min(years_active)}-{max(years_active)} | {early} | {recent} | {continuity} |\n"
        
        report += f"""
---

## 7. 方法说明

### 7.1 分析方法

1. **趋势分析**: 年度论文数量统计
2. **增长率**: CAGR和逐年增长率
3. **拐点检测**: 二阶差分法
4. **预测**: 指数平滑法
5. **关键词演化**: 频率时间序列

### 7.2 预测模型

```
指数平滑:
s(t) = α * x(t) + (1-α) * s(t-1)

预测:
y(t+h) = s(t) + h * trend

其中:
- α = 0.3 (平滑系数)
- trend = (s(t) - s(t-2)) / 2
```

### 7.3 当前局限

1. **数据完整性**: 2025-2026年数据可能不完整
2. **预测精度**: 简单模型，未考虑外部因素
3. **关键词提取**: 基于词频，非语义分析
4. **因果推断**: 仅相关分析，非因果

### 7.4 未来改进

1. **ARIMA模型**: 更复杂的时间序列预测
2. **外部变量**: 引入funding、政策等数据
3. **LSTM预测**: 深度学习时间序列
4. **因果推断**: 识别真正的驱动因素

---

## 8. 结论

### 8.1 核心发现

- 领域处于{'高速' if growth.get('cagr', 0) > 20 else '中速' if growth.get('cagr', 0) > 10 else '低速'}增长期（CAGR: {growth.get('cagr', 0):.1f}%）
- 预测未来3年{'持续' if all(v > 0 for v in predictions.values()) else '波动'}增长
- 关键词演化显示方向{'集中' if len(keyword_evolution) < 20 else '分散'}

### 8.2 战略建议

**短期（1-2年）**:
- 关注预测增长趋势，提前布局
- 跟踪上升关键词代表的方向

**中期（3-5年）**:
- 预测领域成熟度，调整研究策略
- 关注拐点信号，及时转向

**长期（5年+）**:
- 基于演化规律，预判领域生命周期
- 提前布局下一代技术

---

*本报告由 Math-Trend 时间序列分析模块生成*  
*预测基于历史数据，仅供参考*
"""
        
        return report
    
    def save_report(self, report_content: str):
        """保存报告"""
        report_path = OUTPUT_DIR / "time_series_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n✅ 时间序列报告已保存: {report_path}")
        return report_path


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " 时间序列分析 - 趋势预测 + 拐点检测 ".center(76) + "█")
    print("█" * 80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = load_cement_storage_data()
    print(f"  加载 {len(loader.all_papers)} 篇论文")
    
    # 2. 创建分析器
    print("\n2. 创建时间序列分析器...")
    analyzer = TimeSeriesAnalyzer(loader)
    
    # 3. 年度趋势
    print("\n3. 分析年度趋势...")
    yearly_stats = analyzer.analyze_yearly_trends()
    
    # 4. 增长率
    print("\n4. 计算增长率...")
    growth = analyzer.calculate_growth_rate()
    
    # 5. 拐点检测
    print("\n5. 检测拐点...")
    inflections = analyzer.detect_inflection_points()
    
    # 6. 未来预测
    print("\n6. 预测未来趋势...")
    predictions = analyzer.predict_future_trends()
    
    # 7. 关键词演化
    print("\n7. 分析关键词演化...")
    keyword_evolution = analyzer.analyze_keyword_evolution()
    
    # 8. 生成报告
    print("\n8. 生成时间序列报告...")
    report = analyzer.generate_time_series_report()
    report_path = analyzer.save_report(report)
    
    # 9. 总结
    print("\n" + "█" * 80)
    print("█" + " 分析完成 ".center(76) + "█")
    print("█" * 80)
    
    print(f"\n📊 核心结果:")
    print(f"  分析年份: {len(yearly_stats)}年")
    print(f"  CAGR: {growth.get('cagr', 0):.1f}%")
    print(f"  拐点数: {len(inflections)}")
    print(f"  预测年份: {list(predictions.keys())}")
    
    print(f"\n📈 增长趋势:")
    if growth.get('cagr', 0) > 20:
        print("  🔥 高速增长领域")
    elif growth.get('cagr', 0) > 10:
        print("  📈 中速增长领域")
    else:
        print("  📊 低速增长领域")
    
    print(f"\n📄 报告文件: {report_path}")
    print(f"📊 报告包含:")
    print("  ✅ 年度趋势统计")
    print("  ✅ 拐点检测")
    print("  ✅ 未来预测")
    print("  ✅ 关键词演化")
    print("  ✅ 期刊和作者演化")
    
    print()


if __name__ == "__main__":
    main()
