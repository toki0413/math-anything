"""
动态引用网络分析模块

功能：
1. 构建随时间演化的引用网络
2. 分析网络结构动态变化
3. 识别网络演化模式
4. 预测网络未来结构
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from unified_data_loader import UnifiedDataLoader, Paper, load_cement_storage_data

OUTPUT_DIR = Path(__file__).parent / "output" / "dynamic_network"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DynamicNetworkAnalyzer:
    """动态网络分析器"""

    def __init__(self, data_loader: UnifiedDataLoader):
        self.loader = data_loader
        self.papers = data_loader.get_papers_by_confidence("high")
        self.yearly_networks: Dict[int, Dict] = {}

    def build_yearly_networks(self) -> Dict[int, Dict]:
        """
        构建每年的网络快照
        """
        print("\n构建年度网络快照...")

        # 按年分组
        yearly_papers = defaultdict(list)
        for paper in self.papers:
            if paper.year > 0:
                yearly_papers[paper.year].append(paper)

        # 为每年构建网络
        for year in sorted(yearly_papers.keys()):
            papers = yearly_papers[year]

            # 构建网络（简化版）
            network = self._build_single_year_network(papers, year)
            self.yearly_networks[year] = network

        print(f"  构建了 {len(self.yearly_networks)} 个年度网络")
        return self.yearly_networks

    def _build_single_year_network(self, papers: List[Paper], year: int) -> Dict:
        """构建单年网络"""
        # 节点
        nodes = {p.id: {
            "title": p.title,
            "citations": p.citations,
            "authors": p.authors,
            "venue": p.venue
        } for p in papers}

        # 边（基于语义相似度）
        edges = []
        for i, p1 in enumerate(papers):
            for p2 in papers[i+1:]:
                # 计算相似度
                text1 = set(p1.title.lower().split())
                text2 = set(p2.title.lower().split())

                intersection = len(text1 & text2)
                union = len(text1 | text2)
                similarity = intersection / union if union > 0 else 0

                if similarity > 0.2:  # 阈值
                    edges.append({
                        "source": p1.id,
                        "target": p2.id,
                        "weight": similarity
                    })

        # 计算网络指标
        metrics = self._calculate_network_metrics(nodes, edges)

        return {
            "year": year,
            "nodes": nodes,
            "edges": edges,
            "metrics": metrics
        }

    def _calculate_network_metrics(self, nodes: Dict, edges: List[Dict]) -> Dict:
        """计算网络指标"""
        n = len(nodes)
        m = len(edges)

        if n == 0:
            return {"density": 0, "avg_degree": 0, "clustering": 0}

        # 密度
        max_edges = n * (n - 1) / 2
        density = m / max_edges if max_edges > 0 else 0

        # 平均度
        degrees = defaultdict(int)
        for edge in edges:
            degrees[edge["source"]] += 1
            degrees[edge["target"]] += 1

        avg_degree = np.mean(list(degrees.values())) if degrees else 0

        # 聚类系数（简化）
        clustering = self._estimate_clustering(degrees, edges)

        return {
            "node_count": n,
            "edge_count": m,
            "density": density,
            "avg_degree": avg_degree,
            "clustering": clustering
        }

    def _estimate_clustering(self, degrees: Dict[str, int], edges: List[Dict]) -> float:
        """估计聚类系数"""
        if not edges:
            return 0

        # 简化：基于度分布估计
        avg_degree = np.mean(list(degrees.values())) if degrees else 0
        # 近似公式
        clustering = min(avg_degree / max(len(degrees), 1), 1.0)

        return clustering

    def analyze_network_evolution(self) -> Dict[str, List[float]]:
        """分析网络演化趋势"""
        print("\n分析网络演化...")

        if not self.yearly_networks:
            self.build_yearly_networks()

        evolution = {
            "years": [],
            "node_counts": [],
            "edge_counts": [],
            "densities": [],
            "avg_degrees": [],
            "clusterings": []
        }

        for year in sorted(self.yearly_networks.keys()):
            network = self.yearly_networks[year]
            metrics = network["metrics"]

            evolution["years"].append(year)
            evolution["node_counts"].append(metrics["node_count"])
            evolution["edge_counts"].append(metrics["edge_count"])
            evolution["densities"].append(metrics["density"])
            evolution["avg_degrees"].append(metrics["avg_degree"])
            evolution["clusterings"].append(metrics["clustering"])

        return evolution

    def detect_network_patterns(self) -> List[Dict]:
        """检测网络演化模式"""
        print("\n检测网络演化模式...")

        evolution = self.analyze_network_evolution()

        patterns = []

        # 1. 增长模式
        node_counts = evolution["node_counts"]
        if len(node_counts) >= 3:
            if all(node_counts[i] <= node_counts[i+1] for i in range(len(node_counts)-1)):
                patterns.append({
                    "type": "持续增长",
                    "description": "网络规模逐年增加",
                    "confidence": "high"
                })
            elif node_counts[-1] > node_counts[0] * 2:
                patterns.append({
                    "type": "快速增长",
                    "description": "网络规模翻倍以上",
                    "confidence": "high"
                })

        # 2. 密度变化
        densities = evolution["densities"]
        if len(densities) >= 3:
            if densities[-1] < densities[0]:
                patterns.append({
                    "type": "密度稀释",
                    "description": "网络密度下降，领域分散化",
                    "confidence": "medium"
                })
            else:
                patterns.append({
                    "type": "密度增加",
                    "description": "网络密度上升，领域集中化",
                    "confidence": "medium"
                })

        # 3. 聚类系数
        clusterings = evolution["clusterings"]
        if len(clusterings) >= 2:
            if clusterings[-1] > clusterings[0]:
                patterns.append({
                    "type": "社区强化",
                    "description": "社区结构更加明显",
                    "confidence": "medium"
                })

        print(f"  检测到 {len(patterns)} 个演化模式")
        return patterns

    def predict_network_structure(self, years_ahead: int = 3) -> List[Dict]:
        """预测未来网络结构"""
        print(f"\n预测未来{years_ahead}年网络结构...")

        evolution = self.analyze_network_evolution()

        predictions = []
        last_year = evolution["years"][-1]

        # 简单线性预测
        for i in range(1, years_ahead + 1):
            future_year = last_year + i

            # 预测节点数（基于历史趋势）
            if len(evolution["node_counts"]) >= 2:
                growth = (evolution["node_counts"][-1] - evolution["node_counts"][0]) / max(len(evolution["node_counts"]) - 1, 1)
                predicted_nodes = int(evolution["node_counts"][-1] + growth * i)
            else:
                predicted_nodes = evolution["node_counts"][-1] if evolution["node_counts"] else 0

            # 预测密度（假设逐渐稳定）
            if evolution["densities"]:
                recent_density = np.mean(evolution["densities"][-3:])
                predicted_density = max(0.1, recent_density * 0.95)  # 缓慢下降
            else:
                predicted_density = 0.1

            predictions.append({
                "year": future_year,
                "predicted_nodes": max(0, predicted_nodes),
                "predicted_density": predicted_density,
                "predicted_edges": int(predicted_nodes * (predicted_nodes - 1) / 2 * predicted_density) if predicted_nodes > 1 else 0
            })

        print(f"  预测完成: {predictions}")
        return predictions

    def generate_dynamic_report(self, evolution: Dict, patterns: List[Dict], predictions: List[Dict]) -> str:
        """生成动态网络分析报告"""
        print("\n生成动态网络分析报告...")

        report = f"""# 动态引用网络分析报告

**分析方法**: 年度网络快照 + 演化模式识别 + 结构预测  
**网络数量**: {len(self.yearly_networks)}个年度网络  
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. 网络演化概览

### 1.1 年度网络统计

| 年份 | 节点数 | 边数 | 密度 | 平均度 | 聚类系数 |
|-----|-------|------|------|-------|---------|
"""

        for i, year in enumerate(evolution["years"]):
            report += f"| {year} | {evolution['node_counts'][i]} | {evolution['edge_counts'][i]} | {evolution['densities'][i]:.3f} | {evolution['avg_degrees'][i]:.2f} | {evolution['clusterings'][i]:.3f} |\n"

        report += f"""
### 1.2 演化趋势

**节点增长**: {evolution['node_counts'][-1] - evolution['node_counts'][0] if len(evolution['node_counts']) > 1 else 0} ({((evolution['node_counts'][-1] / max(evolution['node_counts'][0], 1) - 1) * 100):.1f}%)
**密度变化**: {evolution['densities'][-1] - evolution['densities'][0] if len(evolution['densities']) > 1 else 0:.3f}
**聚类变化**: {evolution['clusterings'][-1] - evolution['clusterings'][0] if len(evolution['clusterings']) > 1 else 0:.3f}

---

## 2. 演化模式识别

### 2.1 检测到的模式

| 模式 | 描述 | 置信度 |
|-----|------|-------|
"""

        for pattern in patterns:
            report += f"| {pattern['type']} | {pattern['description']} | {pattern['confidence']} |\n"

        if not patterns:
            report += "| 无显著模式 | 网络演化平稳 | - |\n"

        report += f"""
### 2.2 模式解读

"""

        for pattern in patterns:
            report += f"**{pattern['type']}** ({pattern['confidence']}置信度)\n"
            report += f"- {pattern['description']}\n\n"

        report += f"""
---

## 3. 网络结构预测

### 3.1 未来网络结构

| 年份 | 预测节点数 | 预测密度 | 预测边数 |
|-----|-----------|---------|---------|
"""

        for pred in predictions:
            report += f"| {pred['year']} | {pred['predicted_nodes']} | {pred['predicted_density']:.3f} | {pred['predicted_edges']} |\n"

        report += f"""
### 3.2 预测假设

1. **线性增长**: 基于历史平均增长率
2. **密度稳定**: 网络密度缓慢下降趋于稳定
3. **无重大冲击**: 假设无外部重大事件

---

## 4. 网络生命周期

### 4.1 生命周期阶段

基于网络演化特征判断：

```
引入期 → 成长期 → 成熟期 → 衰退期
```

**当前阶段判断**:
- 节点增长: {'快速' if len(evolution['node_counts']) > 1 and evolution['node_counts'][-1] > evolution['node_counts'][0] * 1.5 else '平稳'}
- 密度变化: {'稀释' if len(evolution['densities']) > 1 and evolution['densities'][-1] < evolution['densities'][0] else '稳定'}
- 结论: {'成长期' if len(evolution['node_counts']) > 1 and evolution['node_counts'][-1] > evolution['node_counts'][0] * 1.2 else '成熟期'}

### 4.2 生命周期特征

**成长期特征**:
- 节点快速增长
- 密度可能下降（领域扩展）
- 社区结构逐渐形成

**成熟期特征**:
- 节点增长放缓
- 密度稳定
- 社区结构清晰

---

## 5. 方法说明

### 5.1 网络构建

每年构建一个网络快照：
- 节点: 当年发表的论文
- 边: 基于标题语义相似度

### 5.2 演化分析

比较不同年份的网络指标：
- 规模指标: 节点数、边数
- 结构指标: 密度、平均度
- 社区指标: 聚类系数

### 5.3 预测方法

基于历史趋势的线性外推：
- 假设历史趋势延续
- 考虑网络密度自然衰减

---

## 6. 应用建议

### 6.1 研究策略

- **成长期**: 快速进入，抢占位置
- **成熟期**: 深耕细分，建立壁垒

### 6.2 网络位置

- **中心节点**: 高连接度，影响力大
- **桥梁节点**: 跨社区，创新机会
- **边缘节点**: 新方向，风险高

---

*本报告由 Math-Trend 动态网络分析模块生成*  
*网络基于语义相似度构建，非真实引用网络*
"""

        return report

    def save_report(self, report_content: str):
        """保存报告"""
        report_path = OUTPUT_DIR / "dynamic_network_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n✅ 动态网络报告已保存: {report_path}")
        return report_path


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " 动态引用网络分析 - 演化+预测 ".center(76) + "█")
    print("█" * 80)

    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = load_cement_storage_data()
    print(f"  加载 {len(loader.high_confidence_papers)} 篇高置信度论文")

    # 2. 创建分析器
    print("\n2. 创建动态网络分析器...")
    analyzer = DynamicNetworkAnalyzer(loader)

    # 3. 构建年度网络
    print("\n3. 构建年度网络...")
    yearly_networks = analyzer.build_yearly_networks()

    # 4. 分析演化
    print("\n4. 分析网络演化...")
    evolution = analyzer.analyze_network_evolution()

    # 5. 检测模式
    print("\n5. 检测演化模式...")
    patterns = analyzer.detect_network_patterns()

    # 6. 预测未来
    print("\n6. 预测未来结构...")
    predictions = analyzer.predict_network_structure()

    # 7. 生成报告
    print("\n7. 生成动态网络报告...")
    report = analyzer.generate_dynamic_report(evolution, patterns, predictions)
    report_path = analyzer.save_report(report)

    # 8. 总结
    print("\n" + "█" * 80)
    print("█" + " 分析完成 ".center(76) + "█")
    print("█" * 80)

    print(f"\n📊 核心结果:")
    print(f"  年度网络: {len(yearly_networks)}个")
    print(f"  演化模式: {len(patterns)}个")
    print(f"  未来预测: {len(predictions)}年")

    if patterns:
        print(f"\n🔍 主要模式:")
        for pattern in patterns[:3]:
            print(f"  - {pattern['type']}: {pattern['description']}")

    print(f"\n📄 报告文件: {report_path}")

    print()


if __name__ == "__main__":
    main()
