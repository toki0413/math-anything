"""
报告渲染引擎

统一从 AnalysisResult 生成 Markdown 报告
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List

from core.data_model import AnalysisResult


class ReportRenderer:
    """报告渲染器"""
    
    def __init__(self, template: str = "data_driven"):
        self.template = template
        self.results: Dict[str, AnalysisResult] = {}
    
    def add_result(self, result: AnalysisResult):
        self.results[result.module_name] = result
    
    def render(self, config=None) -> str:
        domain = config.domain_name if config else "研究领域"
        
        report = f"""# {domain} - 数据驱动研究报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**分析模块**: {len(self.results)}个
**成功模块**: {sum(1 for r in self.results.values() if r.success)}个

---

"""
        
        # 1. 数据概览
        report += self._render_statistics()
        
        # 2. 跨领域迁移
        report += self._render_cross_domain()
        
        # 3. 期刊排名
        report += self._render_journal_ranking()
        
        # 4. 主题分析
        report += self._render_topic_modeling()
        
        # 5. 网络分析
        report += self._render_citation_network()
        
        # 6. 时间趋势
        report += self._render_time_series()
        
        # 7. 跨领域对比
        report += self._render_cross_field_comparison()
        
        report += "---\n\n"
        report += "*本报告基于公开学术文献数据自动生成*\n"
        report += "*所有结论均为数据中的模式描述，需结合领域知识解读*\n"
        
        return report
    
    def _render_statistics(self) -> str:
        r = self.results.get("basic.statistics")
        if not r or not r.success:
            return ""
        
        s = r.data.get("scale", {})
        section = f"""## 1. 数据概览

| 指标 | 数值 |
|-----|------|
| 总论文数 | {s.get('total', 'N/A')}篇 |
| 高置信度 | {s.get('high_confidence', 'N/A')}篇 |
| 时间跨度 | {s.get('year_range', 'N/A')} |
| 期刊数 | {s.get('unique_venues', 'N/A')}个 |
| 平均被引 | {s.get('avg_citations', 'N/A')}次 |

### 年度分布

"""
        yearly = r.data.get("yearly_distribution", {})
        for year, count in sorted(yearly.items()):
            bar = '█' * min(count, 50)
            section += f"{year}: {bar} ({count}篇)\n"
        
        # Top关键词
        top_kw = r.data.get("top_keywords", {})
        if top_kw:
            section += "\n### Top关键词\n\n"
            for kw, count in sorted(top_kw.items(), key=lambda x: x[1], reverse=True)[:10]:
                section += f"- **{kw}**: {count}篇\n"
        
        # 高被引论文
        top_cited = r.data.get("top_10_cited", [])
        if top_cited:
            section += "\n### Top 10 高被引论文\n\n"
            for i, p in enumerate(top_cited, 1):
                section += f"{i}. [{p.get('year', 'N/A')}] {p.get('title', 'N/A')[:80]}... ({p.get('citations', 0)}次)\n"
        
        section += "\n---\n\n"
        return section
    
    def _render_cross_domain(self) -> str:
        r = self.results.get("basic.cross_domain")
        if not r or not r.success:
            return ""
        
        section = f"""## 2. 跨领域知识迁移

检测到 {r.data.get('total_transfers', 0)} 个迁移事件（高置信度: {r.data.get('high_confidence_transfers', 0)}个）

### 迁移来源分布

"""
        source_dist = r.data.get("source_distribution", {})
        for source, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True):
            section += f"- **{source}**: {count}篇论文\n"
        
        transfers = r.data.get("transfers", [])
        if transfers:
            section += "\n### 迁移详情\n\n"
            section += "| 来源领域 | 论文数 | 首次出现 | 峰值年 | 置信度 |\n"
            section += "|---------|--------|---------|--------|--------|\n"
            for t in transfers[:8]:
                section += f"| {t['source_domain']} | {t['paper_count']} | {t['first_appearance']} | {t['peak_year']} | {t['confidence']} |\n"
        
        section += "\n---\n\n"
        return section
    
    def _render_journal_ranking(self) -> str:
        r = self.results.get("basic.journal_ranking")
        if not r or not r.success:
            return ""
        
        section = f"""## 3. 期刊排名

共 {r.data.get('total_venues', 0)} 个期刊（≥2篇论文）

"""
        rankings = r.data.get("rankings", [])
        if rankings:
            section += "| 排名 | 期刊 | 论文数 | 篇均被引 | 基尼系数 | 分级 |\n"
            section += "|-----|------|--------|---------|---------|------|\n"
            for i, j in enumerate(rankings[:10], 1):
                section += f"| {i} | {j['venue'][:40]} | {j['papers']} | {j['avg_citations']} | {j['gini']} | {j['tier']} |\n"
        
        tier_dist = r.data.get("tier_distribution", {})
        if tier_dist:
            section += f"\n分级分布: {tier_dist}\n"
        
        section += "\n---\n\n"
        return section
    
    def _render_topic_modeling(self) -> str:
        r = self.results.get("advanced.topic_modeling")
        if not r or not r.success:
            return ""
        
        section = "## 4. 主题分析\n\n"
        
        topics = r.data.get("topics", [])
        if topics:
            section += "### 主题聚类\n\n"
            for i, t in enumerate(topics[:6], 1):
                members = ', '.join(t['members'][:5])
                section += f"{i}. **{t['core_keyword']}** ({t['total_papers']}篇): {members}\n"
        
        emerging = r.data.get("emerging_topics", [])
        if emerging:
            section += "\n### 新兴主题\n\n"
            for e in emerging[:5]:
                section += f"- **{e['keyword']}**: {e['status']}（近期{e['recent_count']}篇）\n"
        
        section += "\n---\n\n"
        return section
    
    def _render_citation_network(self) -> str:
        r = self.results.get("advanced.citation_network")
        if not r or not r.success:
            return ""
        
        stats = r.data.get("network_stats", {})
        section = f"""## 5. 网络分析

| 指标 | 数值 |
|-----|------|
| 节点数 | {stats.get('nodes', 0)} |
| 边数 | {stats.get('edges', 0)} |
| 密度 | {stats.get('density', 0)} |
| 社区数 | {r.data.get('communities', 0)} |

"""
        top_pr = r.data.get("top_pagerank", [])
        if top_pr:
            section += "### Top PageRank节点\n\n"
            for n in top_pr[:5]:
                section += f"- {n['node']}: {n['score']}\n"
        
        section += "\n---\n\n"
        return section
    
    def _render_time_series(self) -> str:
        r = self.results.get("advanced.time_series")
        if not r or not r.success:
            return ""
        
        section = f"""## 6. 时间趋势

- **CAGR**: {r.data.get('cagr_percent', 'N/A')}%
- **平均增长率**: {r.data.get('avg_growth_rate', 'N/A')}%

"""
        predictions = r.data.get("predictions", {})
        if predictions:
            section += "### 预测（线性外推）\n\n"
            for year, count in sorted(predictions.items()):
                section += f"- {year}: {count}篇\n"
        
        inflections = r.data.get("inflection_points", [])
        if inflections:
            section += "\n### 拐点\n\n"
            for pt in inflections:
                section += f"- {pt['year']}年: {pt['type']} ({pt['count']}篇)\n"
        
        section += "\n---\n\n"
        return section
    
    def _render_cross_field_comparison(self) -> str:
        r = self.results.get("advanced.cross_field_comparison")
        if not r or not r.success:
            return ""
        
        section = "## 7. 跨领域对比\n\n"
        
        table = r.data.get("comparison_table", [])
        if table:
            section += "| 领域 | 论文数 | 篇均被引 | 最高被引 | 备注 |\n"
            section += "|-----|--------|---------|---------|------|\n"
            for t in table:
                marker = "🎯 目标领域" if t.get('is_target') else ""
                section += f"| {t['field']} | {t['papers']} | {t['avg_citations']} | {t['max_citations']} | {marker} |\n"
        
        growth = r.data.get("growth_comparison", {})
        if growth:
            section += "\n### 增长对比\n\n"
            target = growth.get('target', {})
            section += f"- **{target.get('name', '目标')}**: CAGR {target.get('cagr', 0)}%, 近3年均 {target.get('recent_3yr_avg', 0)}篇\n"
            
            for key, comp in growth.get('comparisons', {}).items():
                section += f"- **{comp.get('name', key)}**: CAGR {comp.get('cagr', 0)}%, 近3年均 {comp.get('recent_3yr_avg', 0)}篇\n"
        
        section += "\n---\n\n"
        return section
    
    def save(self, output_path: Path, config=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.render(config)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ 报告已保存: {output_path}")
        return output_path
