"""
Actionable Insights Report Generator
生成有深度、可操作的研究分析报告

原则：
1. 只说有数据支撑的话
2. 标注推测，不伪装成事实
3. 不做无依据的定量判断（资金、市场份额等）
4. 识别模式，不编造原因
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

from unified_data_loader import load_cement_storage_data, Paper

OUTPUT_DIR = Path(__file__).parent / "output" / "actionable_insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ActionableInsightAnalyzer:
    """
    可操作的洞察分析器
    
    严格遵循：只报告数据中的模式，不编造解释
    """
    
    def __init__(self):
        self.loader = load_cement_storage_data()
        self.high_papers = self.loader.high_confidence_papers
        self.all_papers = self.loader.all_papers
        
        # 验证作者字段格式
        self._validate_author_data()
    
    def _validate_author_data(self):
        """检查作者字段是否干净"""
        sample = self.high_papers[:5]
        for p in sample:
            for a in p.authors[:2]:
                if len(a) > 50 or any(x in a for x in ['University', 'Institute', 'College', 'School', 'Department', 'Lab']):
                    # 作者字段包含机构信息，统计时需要过滤
                    self._author_field_dirty = True
                    return
        self._author_field_dirty = False
        
    def analyze_keyword_trends(self) -> Dict:
        """
        关键词趋势分析 - 基于实际数据
        
        返回：哪些词在增长，哪些在衰退
        """
        print("\n分析关键词趋势...")
        
        # 按3年窗口统计
        core_terms = [
            'supercapacitor', 'battery', 'electrode', 'electrolyte',
            'carbon nanotube', 'CNT', 'graphene', 'activated carbon',
            'PEDOT', 'conductive', 'multifunctional', 'self-sensing',
            'compressive strength', 'capacitance'
        ]
        
        windows = {
            '2018-2020': range(2018, 2021),
            '2021-2023': range(2021, 2024),
            '2024-2026': range(2024, 2027)
        }
        
        trends = {}
        for term in core_terms:
            counts = {}
            for label, year_range in windows.items():
                count = sum(1 for p in self.all_papers 
                           if p.year in year_range 
                           and term.lower() in f"{p.title} {p.abstract}".lower())
                counts[label] = count
            trends[term] = counts
        
        # 识别增长/衰退趋势
        growing = []
        declining = []
        stable = []
        
        for term, counts in trends.items():
            early = counts.get('2018-2020', 0)
            late = counts.get('2024-2026', 0)
            
            if early == 0 and late > 0:
                growing.append((term, '新出现', late))
            elif late > early * 1.5:
                growing.append((term, '增长', late))
            elif late < early * 0.5 and early > 0:
                declining.append((term, '衰退', late))
            else:
                stable.append((term, '稳定', late))
        
        print(f"  增长: {len(growing)}, 稳定: {len(stable)}, 衰退: {len(declining)}")
        return {
            'trends': trends,
            'growing': growing,
            'declining': declining,
            'stable': stable
        }
    
    def analyze_research_patterns(self) -> Dict:
        """
        识别数据中的真实模式
        
        不做推测，只报告观察到的现象
        """
        print("\n分析研究模式...")
        
        patterns = []
        
        # 模式1：论文集中度
        journal_counts = Counter(p.venue for p in self.high_papers if p.venue)
        top_5_journals = journal_counts.most_common(5)
        total_in_top5 = sum(c for _, c in top_5_journals)
        concentration = total_in_top5 / len(self.high_papers) * 100
        
        patterns.append({
            'type': '期刊集中度',
            'observation': f'Top 5期刊发表{total_in_top5}篇，占总量的{concentration:.1f}%',
            'journals': top_5_journals,
            'implication': '领域期刊分布较为集中或分散（需结合领域特点判断）'
        })
        
        # 模式2：年度增长
        yearly = Counter(p.year for p in self.all_papers if p.year > 0)
        years_sorted = sorted(yearly.keys())
        if len(years_sorted) >= 2:
            first_year = years_sorted[0]
            last_year = years_sorted[-1]
            first_count = yearly[first_year]
            last_count = yearly[last_year]
            
            patterns.append({
                'type': '时间趋势',
                'observation': f'{first_year}年{first_count}篇 → {last_year}年{last_count}篇',
                'data': dict(yearly),
                'implication': '论文数量呈增长/平稳趋势' if last_count > first_count else '论文数量呈下降趋势'
            })
        
        # 模式3：高被引论文的共同特征
        sorted_by_citation = sorted(self.high_papers, key=lambda x: x.citations, reverse=True)
        top_10 = sorted_by_citation[:10]
        
        # 统计top10的共同关键词
        top10_text = ' '.join(f"{p.title} {p.abstract}".lower() for p in top_10)
        indicators = ['review', 'prototype', 'nanocomposite', 'cementitious', 'electrochemical']
        common_features = [(ind, top10_text.count(ind)) for ind in indicators if top10_text.count(ind) > 0]
        
        if common_features:
            patterns.append({
                'type': '高被引论文特征',
                'observation': f'Top 10高被引论文（被引{sorted_by_citation[0].citations}-{sorted_by_citation[9].citations}次）的共同特征：',
                'features': common_features,
                'top_paper_title': top_10[0].title[:100],
                'implication': '高被引论文倾向于关注这些方向'
            })
        
        # 模式4：研究主题演变
        early_papers = [p for p in self.all_papers if p.year <= 2020]
        late_papers = [p for p in self.all_papers if p.year >= 2024]
        
        early_text = ' '.join(f"{p.title} {p.abstract}".lower() for p in early_papers)
        late_text = ' '.join(f"{p.title} {p.abstract}".lower() for p in late_papers)
        
        shift_terms = []
        for term in ['machine learning', 'AI', 'deep learning', 'sustainability', 'carbon neutral', 'building integrated']:
            early_count = early_text.count(term)
            late_count = late_text.count(term)
            if early_count == 0 and late_count > 0:
                shift_terms.append((term, '新出现'))
            elif late_count > early_count * 2:
                shift_terms.append((term, f'{early_count}→{late_count}'))
        
        if shift_terms:
            patterns.append({
                'type': '主题演变',
                'observation': '2024年后新出现或显著增长的术语',
                'shifts': shift_terms,
                'implication': '领域关注点可能正在转向这些方向'
            })
        
        print(f"  识别 {len(patterns)} 个模式")
        return {'patterns': patterns}
    
    def analyze_collaboration_targets(self) -> List[Dict]:
        """
        合作目标分析 - 基于实际数据
        
        注意：作者字段可能包含机构信息，结果需要人工验证
        """
        print("\n分析合作目标...")
        
        # 只统计前3作者，且过滤掉明显的机构名
        author_citations = defaultdict(lambda: {'citations': 0, 'papers': 0})
        
        for paper in self.high_papers:
            for author in paper.authors[:3]:
                # 过滤机构名
                if any(x in author for x in ['University', 'Institute', 'College', 'School', 'Department', 'Lab', 'Academy']):
                    continue
                if len(author) > 60:  # 过长的通常是机构
                    continue
                    
                author_citations[author]['citations'] += paper.citations
                author_citations[author]['papers'] += 1
        
        targets = []
        for author, data in author_citations.items():
            if data['papers'] >= 2:  # 至少2篇
                targets.append({
                    'author': author,
                    'papers': data['papers'],
                    'citations': data['citations'],
                    'avg_citations': data['citations'] / data['papers']
                })
        
        targets.sort(key=lambda x: x['citations'], reverse=True)
        
        # 添加验证警告
        warning = "注：以下统计基于数据中的作者字段，可能存在同名不同人的情况，需人工验证"
        
        print(f"  识别 {len(targets)} 个潜在合作目标（需人工验证）")
        return {'targets': targets[:10], 'warning': warning}
    
    def generate_report(self) -> str:
        """生成报告"""
        print("\n生成报告...")
        
        keyword_trends = self.analyze_keyword_trends()
        patterns = self.analyze_research_patterns()
        collaborators = self.analyze_collaboration_targets()
        
        report = f"""# 水泥基电化学储能 - 数据驱动研究分析报告

**报告原则**: 只报告数据中的模式，不做无依据推测  
**数据基础**: {len(self.high_papers)}篇高置信度 + {len(self.all_papers) - len(self.high_papers)}篇中低置信度  
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. 数据事实（可直接验证）

### 1.1 规模

| 指标 | 数值 |
|-----|------|
| 总论文数 | {len(self.all_papers)}篇 |
| 高置信度 | {len(self.high_papers)}篇 |
| 时间跨度 | {min(p.year for p in self.all_papers if p.year > 0)}-{max(p.year for p in self.all_papers if p.year > 0)} |
| 期刊数 | {len(set(p.venue for p in self.all_papers if p.venue))}个 |
| 最高被引 | {max(p.citations for p in self.all_papers)}次 |

### 1.2 年度分布

"""
        yearly = Counter(p.year for p in self.all_papers if p.year > 0)
        for year in sorted(yearly.keys()):
            bar = '█' * yearly[year]
            report += f"{year}: {bar} ({yearly[year]}篇)\n"
        
        report += f"""
---

## 2. 观察到的模式（非推测）

"""
        
        for p in patterns['patterns']:
            report += f"### {p['type']}\n\n"
            report += f"**数据**: {p['observation']}\n\n"
            
            if 'journals' in p:
                report += "| 期刊 | 篇数 |\n|-----|------|\n"
                for j, c in p['journals']:
                    report += f"| {j} | {c} |\n"
                report += "\n"
            
            if 'features' in p:
                for feat, count in p['features']:
                    report += f"- 含'{feat}'的高被引论文: {count}次提及\n"
                report += "\n"
            
            if 'shifts' in p:
                for term, change in p['shifts']:
                    report += f"- {term}: {change}\n"
                report += "\n"
            
            report += f"**解读**: {p['implication']}\n\n"
            report += "---\n\n"
        
        report += f"""
## 3. 关键词趋势

### 3.1 增长中的术语

"""
        for term, trend, count in keyword_trends['growing']:
            report += f"- **{term}**: {trend}（最近窗口出现{count}次）\n"
        
        report += f"""
### 3.2 稳定的术语

"""
        for term, trend, count in keyword_trends['stable']:
            report += f"- **{term}**: {trend}（最近窗口出现{count}次）\n"
        
        if keyword_trends['declining']:
            report += f"""
### 3.3 衰退中的术语

"""
            for term, trend, count in keyword_trends['declining']:
                report += f"- **{term}**: {trend}（最近窗口出现{count}次）\n"
        
        report += f"""
---

## 4. 潜在合作者（需人工验证）

{collaborators['warning']}

"""
        
        for i, t in enumerate(collaborators['targets'][:10], 1):
            report += f"{i}. {t['author']}: {t['papers']}篇, {t['citations']}次被引, 篇均{t['avg_citations']:.1f}次\n"
        
        report += f"""
---

## 5. 报告局限（诚实声明）

### 本报告**能**做的：
- ✅ 统计论文数量、引用数、关键词频率
- ✅ 识别时间趋势（增长/稳定/衰退）
- ✅ 发现数据中的模式（集中度、共现等）

### 本报告**不能**做的：
- ❌ 判断市场大小或市场份额（需要商业数据）
- ❌ 推荐具体投资金额（没有财务模型）
- ❌ 评估技术成熟度TRL（需要专家判断）
- ❌ 识别"研究空白"（没有数据≠是空白，可能是不值得研究）
- ❌ 预测未来趋势（外推不等于预测）
- ❌ 推荐合作对象（需要人工验证身份和方向匹配度）

### 数据来源与质量：
- 数据来自OpenAlex、CrossRef、Semantic Scholar
- 置信度分层基于期刊质量、被引数、作者经验等指标
- 作者字段可能包含机构信息，统计结果需人工核实
- 关键词匹配基于字符串搜索，非语义理解

---

*本报告基于公开学术文献数据生成*
*所有结论均为数据中的模式描述，非预测或建议*
*决策需结合专家意见和商业数据*
"""
        
        # 保存
        report_path = OUTPUT_DIR / "data_driven_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ 数据驱动报告已保存: {report_path}")
        print(f"  文件大小: {report_path.stat().st_size:,} bytes")
        
        return report


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " 数据驱动研究分析（严格反幻觉） ".center(76) + "█")
    print("█" * 80)
    
    analyzer = ActionableInsightAnalyzer()
    analyzer.generate_report()
    
    print("\n" + "█" * 80)
    print("█" + " 分析完成 ".center(76) + "█")
    print("█" * 80)


def main_legacy():
    """旧版：可操作洞察（保留但不默认使用）"""
    print("\n" + "█" * 80)
    print("█" + " 可操作研究洞察分析 ".center(76) + "█")
    print("█" * 80)
    
    analyzer = ActionableInsightAnalyzer()
    analyzer.generate_report()
    
    print("\n" + "█" * 80)
    print("█" + " 分析完成 ".center(76) + "█")
    print("█" * 80)


if __name__ == "__main__":
    main()
