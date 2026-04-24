"""
跨领域知识迁移检测模块

识别和量化知识从一个领域向另一个领域的迁移过程，
包括概念借用、方法迁移、理论跨域应用等
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re
from difflib import SequenceMatcher


@dataclass
class KnowledgeTransfer:
    """知识迁移事件"""
    source_domain: str
    target_domain: str
    concept: str
    transfer_type: str  # method/theory/tool/concept
    strength: float
    first_occurrence: int  # 年份
    peak_year: int
    adoption_curve: List[float] = field(default_factory=list)
    key_papers: List[str] = field(default_factory=list)
    confidence: str = "medium"


class CrossDomainTransferDetector:
    """跨领域知识迁移检测器"""
    
    def __init__(self):
        self.domain_terms = {}
        self.transfer_history = []
        
    def extract_domain_signatures(
        self,
        papers: List[Dict],
        domain_name: str
    ) -> Dict[str, float]:
        """
        提取领域特征签名（关键词TF-IDF权重）
        
        Args:
            papers: 领域论文列表
            domain_name: 领域名称
            
        Returns:
            术语 -> 权重 的映射
        """
        # 统计词频
        term_freq = defaultdict(int)
        doc_count = len(papers)
        
        for paper in papers:
            # 从标题和摘要提取术语
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            terms = self._extract_terms(text)
            
            for term in set(terms):  # 每篇文档只计一次
                term_freq[term] += 1
        
        # 计算TF-IDF权重（简化版）
        signatures = {}
        for term, freq in term_freq.items():
            tf = freq / doc_count
            # 使用领域特异性作为IDF代理
            signatures[term] = tf
        
        # 保留前100个特征词
        sorted_terms = sorted(signatures.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_terms[:100])
    
    def _extract_terms(self, text: str) -> List[str]:
        """从文本中提取术语"""
        # 简化的术语提取：保留2-4个词的短语
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        terms = []
        
        # 单个词（长度>4）
        terms.extend([w for w in words if len(w) > 4])
        
        # 2-3词短语
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                terms.append(phrase)
        
        return terms
    
    def detect_concept_transfer(
        self,
        source_papers: List[Dict],
        target_papers: List[Dict],
        source_domain: str,
        target_domain: str,
        year_from: int = 2015,
        year_to: int = 2026
    ) -> List[KnowledgeTransfer]:
        """
        检测概念从源领域向目标领域的迁移
        
        Args:
            source_papers: 源领域论文
            target_papers: 目标领域论文
            source_domain: 源领域名
            target_domain: 目标领域名
            
        Returns:
            检测到的知识迁移列表
        """
        transfers = []
        
        # 提取源领域特征术语
        source_signatures = self.extract_domain_signatures(source_papers, source_domain)
        source_terms = set(source_signatures.keys())
        
        # 按年份分析目标领域
        target_by_year = defaultdict(list)
        for paper in target_papers:
            year = paper.get('year', 2020)
            target_by_year[year].append(paper)
        
        # 追踪每个源术语在目标领域的出现
        for term in source_terms:
            transfer = self._track_term_migration(
                term,
                target_by_year,
                source_domain,
                target_domain,
                year_from,
                year_to
            )
            
            if transfer and transfer.strength > 0.3:
                transfers.append(transfer)
        
        # 按强度排序
        transfers.sort(key=lambda x: x.strength, reverse=True)
        return transfers[:20]  # 返回前20个
    
    def _track_term_migration(
        self,
        term: str,
        target_by_year: Dict[int, List[Dict]],
        source_domain: str,
        target_domain: str,
        year_from: int,
        year_to: int
    ) -> Optional[KnowledgeTransfer]:
        """追踪单个术语的跨域迁移"""
        
        yearly_freq = []
        first_year = None
        peak_year = year_from
        peak_freq = 0
        key_papers = []
        
        for year in range(year_from, year_to + 1):
            papers = target_by_year.get(year, [])
            freq = 0
            
            for paper in papers:
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
                if term in text:
                    freq += 1
                    if len(key_papers) < 5:
                        key_papers.append(paper.get('id', 'unknown'))
            
            yearly_freq.append(freq)
            
            if freq > 0 and first_year is None:
                first_year = year
            
            if freq > peak_freq:
                peak_freq = freq
                peak_year = year
        
        # 判断是否为有效迁移
        if first_year is None or first_year == year_from:
            return None
        
        # 计算迁移强度
        total_papers = sum(len(target_by_year.get(y, [])) for y in range(year_from, year_to + 1))
        strength = peak_freq / total_papers if total_papers > 0 else 0
        
        # 判断迁移类型
        transfer_type = self._classify_transfer_type(term, source_domain, target_domain)
        
        # 判断置信度
        if strength > 0.1 and peak_year - first_year >= 2:
            confidence = "high"
        elif strength > 0.05:
            confidence = "medium"
        else:
            confidence = "low"
        
        return KnowledgeTransfer(
            source_domain=source_domain,
            target_domain=target_domain,
            concept=term,
            transfer_type=transfer_type,
            strength=strength,
            first_occurrence=first_year,
            peak_year=peak_year,
            adoption_curve=yearly_freq,
            key_papers=key_papers,
            confidence=confidence
        )
    
    def _classify_transfer_type(
        self,
        term: str,
        source_domain: str,
        target_domain: str
    ) -> str:
        """分类知识迁移类型"""
        # 基于术语特征分类
        method_keywords = ['algorithm', 'method', 'technique', 'approach', 'model']
        theory_keywords = ['theory', 'principle', 'law', 'framework', 'paradigm']
        tool_keywords = ['software', 'tool', 'platform', 'device', 'instrument']
        
        term_lower = term.lower()
        
        if any(kw in term_lower for kw in method_keywords):
            return "method"
        elif any(kw in term_lower for kw in theory_keywords):
            return "theory"
        elif any(kw in term_lower for kw in tool_keywords):
            return "tool"
        else:
            return "concept"
    
    def calculate_semantic_bridge(
        self,
        domain1_signatures: Dict[str, float],
        domain2_signatures: Dict[str, float]
    ) -> Dict[str, any]:
        """
        计算两个领域之间的语义桥梁
        
        找出连接两个领域的共同概念和过渡术语
        
        Returns:
            语义桥梁分析结果
        """
        # 找出重叠术语
        terms1 = set(domain1_signatures.keys())
        terms2 = set(domain2_signatures.keys())
        
        overlap = terms1 & terms2
        unique_to_1 = terms1 - terms2
        unique_to_2 = terms2 - terms1
        
        # 计算领域相似度（Jaccard + 余弦相似度）
        jaccard = len(overlap) / len(terms1 | terms2) if (terms1 | terms2) else 0
        
        # 余弦相似度
        all_terms = terms1 | terms2
        vec1 = np.array([domain1_signatures.get(t, 0) for t in all_terms])
        vec2 = np.array([domain2_signatures.get(t, 0) for t in all_terms])
        
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        
        # 识别桥梁术语（在重叠中权重较高的）
        bridge_terms = sorted(
            [(t, domain1_signatures.get(t, 0) + domain2_signatures.get(t, 0)) 
             for t in overlap],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "similarity_jaccard": jaccard,
            "similarity_cosine": cosine_sim,
            "overlap_size": len(overlap),
            "bridge_terms": bridge_terms,
            "unique_terms_domain1": list(unique_to_1)[:10],
            "unique_terms_domain2": list(unique_to_2)[:10],
            "bridge_strength": (jaccard + cosine_sim) / 2
        }
    
    def identify_migration_paths(
        self,
        transfers: List[KnowledgeTransfer]
    ) -> List[Dict]:
        """
        识别知识迁移的多跳路径
        
        例如：A领域 → B领域 → C领域
        
        Returns:
            迁移路径列表
        """
        # 构建迁移图
        migration_graph = defaultdict(list)
        
        for transfer in transfers:
            migration_graph[transfer.source_domain].append({
                "target": transfer.target_domain,
                "concept": transfer.concept,
                "strength": transfer.strength,
                "year": transfer.first_occurrence
            })
        
        # 寻找2跳路径
        paths = []
        
        for source, first_hops in migration_graph.items():
            for hop1 in first_hops:
                intermediate = hop1["target"]
                
                if intermediate in migration_graph:
                    for hop2 in migration_graph[intermediate]:
                        if hop2["target"] != source:  # 避免循环
                            paths.append({
                                "path": f"{source} → {intermediate} → {hop2['target']}",
                                "hops": [
                                    {"from": source, "to": intermediate, 
                                     "concept": hop1["concept"], "year": hop1["year"]},
                                    {"from": intermediate, "to": hop2["target"],
                                     "concept": hop2["concept"], "year": hop2["year"]}
                                ],
                                "total_strength": hop1["strength"] * hop2["strength"],
                                "time_span": hop2["year"] - hop1["year"]
                            })
        
        # 按强度排序
        paths.sort(key=lambda x: x["total_strength"], reverse=True)
        return paths[:10]
    
    def analyze_adoption_pattern(
        self,
        transfer: KnowledgeTransfer
    ) -> Dict[str, any]:
        """
        分析概念在目标领域的采纳模式
        
        识别采纳曲线类型：
        - 爆发式（viral）：快速达到高峰
        - 渐进式（gradual）：稳定增长
        - 波动式（fluctuating）：不稳定
        - 衰退式（declining）：早期高峰后下降
        """
        curve = transfer.adoption_curve
        
        if not curve or sum(curve) == 0:
            return {"pattern": "unknown"}
        
        # 计算增长速率
        growth_rates = []
        for i in range(1, len(curve)):
            if curve[i-1] > 0:
                rate = (curve[i] - curve[i-1]) / curve[i-1]
                growth_rates.append(rate)
        
        avg_growth = np.mean(growth_rates) if growth_rates else 0
        max_value = max(curve)
        max_index = curve.index(max_value)
        total = sum(curve)
        
        # 判断模式
        if max_index <= 3 and max_value > total * 0.5:
            pattern = "爆发式 (viral)"
            description = "概念被迅速采纳，早期即达到高峰"
        elif avg_growth > 0.1:
            pattern = "渐进式 (gradual)"
            description = "稳定增长，持续获得关注"
        elif max_value < total * 0.2:
            pattern = "波动式 (fluctuating)"
            description = "采纳不稳定，受外部因素影响"
        elif max_index < len(curve) / 2:
            pattern = "衰退式 (declining)"
            description = "早期高峰后逐渐衰退"
        else:
            pattern = "稳定式 (steady)"
            description = "持续稳定使用"
        
        return {
            "pattern": pattern,
            "description": description,
            "peak_year_relative": max_index,
            "growth_rate": avg_growth,
            "longevity": len([c for c in curve if c > 0])
        }
    
    def detect_emerging_bridges(
        self,
        domain_pairs: List[Tuple[str, str, List[Dict], List[Dict]]],
        recent_years: int = 3
    ) -> List[Dict]:
        """
        检测新兴的领域间桥梁（近期快速增长的跨领域连接）
        
        Args:
            domain_pairs: [(domain1_name, domain2_name, papers1, papers2), ...]
            recent_years: 近期年数
            
        Returns:
            新兴桥梁列表
        """
        emerging = []
        
        for domain1, domain2, papers1, papers2 in domain_pairs:
            # 只分析近期论文
            recent_papers1 = [p for p in papers1 
                            if p.get('year', 2020) >= 2024 - recent_years]
            recent_papers2 = [p for p in papers2 
                            if p.get('year', 2020) >= 2024 - recent_years]
            
            if len(recent_papers1) < 10 or len(recent_papers2) < 10:
                continue
            
            # 计算桥梁强度
            sig1 = self.extract_domain_signatures(recent_papers1, domain1)
            sig2 = self.extract_domain_signatures(recent_papers2, domain2)
            
            bridge = self.calculate_semantic_bridge(sig1, sig2)
            
            if bridge["bridge_strength"] > 0.2:  # 阈值
                emerging.append({
                    "domain_pair": (domain1, domain2),
                    "bridge_strength": bridge["bridge_strength"],
                    "bridge_terms": bridge["bridge_terms"][:5],
                    "similarity": bridge["similarity_cosine"],
                    "opportunity": "高" if bridge["bridge_strength"] > 0.4 else "中"
                })
        
        # 按桥梁强度排序
        emerging.sort(key=lambda x: x["bridge_strength"], reverse=True)
        return emerging
    
    def generate_transfer_report(
        self,
        transfers: List[KnowledgeTransfer],
        source_domain: str,
        target_domain: str
    ) -> str:
        """生成知识迁移分析报告"""
        
        report = f"""# 跨领域知识迁移分析报告

## {source_domain} → {target_domain}

### 迁移概览
- 检测到的迁移事件: {len(transfers)}个
- 高置信度迁移: {len([t for t in transfers if t.confidence == 'high'])}个
- 平均迁移强度: {np.mean([t.strength for t in transfers]):.3f}

### 主要迁移概念（Top 10）

"""
        
        for i, transfer in enumerate(transfers[:10], 1):
            adoption = self.analyze_adoption_pattern(transfer)
            
            report += f"""{i}. **{transfer.concept}**
   - 类型: {transfer.transfer_type}
   - 首次出现: {transfer.first_occurrence}年
   - 高峰期: {transfer.peak_year}年
   - 强度: {transfer.strength:.3f}
   - 置信度: {transfer.confidence}
   - 采纳模式: {adoption['pattern']}
   
"""
        
        # 添加迁移路径分析
        paths = self.identify_migration_paths(transfers)
        if paths:
            report += "\n### 多跳迁移路径\n\n"
            for path in paths[:5]:
                report += f"- {path['path']} (强度: {path['total_strength']:.3f})\n"
        
        report += """
### 建议

**对目标领域研究者：**
- 关注高置信度的迁移概念，可能带来方法论突破
- 分析采纳模式，选择合适时机引入新概念

**对源领域研究者：**
- 识别正在向外迁移的核心概念
- 考虑跨领域合作机会
"""
        
        return report


def example_usage():
    """使用示例"""
    
    # 模拟数据：电池领域 → 水泥领域
    battery_papers = [
        {"id": "b1", "title": "Lithium ion battery electrode design", 
         "abstract": "electrochemical impedance spectroscopy method", "year": 2018},
        {"id": "b2", "title": "Solid electrolyte interphase formation", 
         "abstract": "cyclic voltammetry analysis technique", "year": 2019},
        {"id": "b3", "title": "Battery management system algorithm", 
         "abstract": "machine learning prediction model", "year": 2020},
    ]
    
    cement_papers = [
        {"id": "c1", "title": "Cement-based supercapacitor", 
         "abstract": "electrochemical impedance spectroscopy characterization", "year": 2020},
        {"id": "c2", "title": "Conductive cement composite", 
         "abstract": "cyclic voltammetry testing method", "year": 2021},
        {"id": "c3", "title": "Energy storage concrete", 
         "abstract": "machine learning optimization", "year": 2022},
        {"id": "c4", "title": "Carbon nanotube cement", 
         "abstract": "electrochemical analysis", "year": 2023},
    ]
    
    detector = CrossDomainTransferDetector()
    
    print("跨领域知识迁移检测")
    print("=" * 60)
    print("\n源领域: 电池技术")
    print("目标领域: 水泥基储能")
    
    # 检测迁移
    transfers = detector.detect_concept_transfer(
        battery_papers,
        cement_papers,
        "电池技术",
        "水泥基储能",
        year_from=2018,
        year_to=2023
    )
    
    print(f"\n检测到 {len(transfers)} 个知识迁移事件:\n")
    
    for i, transfer in enumerate(transfers[:5], 1):
        adoption = detector.analyze_adoption_pattern(transfer)
        print(f"{i}. {transfer.concept}")
        print(f"   类型: {transfer.transfer_type}")
        print(f"   迁移时间: {transfer.first_occurrence}年")
        print(f"   强度: {transfer.strength:.3f}")
        print(f"   采纳模式: {adoption['pattern']}")
        print()
    
    # 计算语义桥梁
    print("=" * 60)
    print("\n语义桥梁分析")
    
    sig_battery = detector.extract_domain_signatures(battery_papers, "电池")
    sig_cement = detector.extract_domain_signatures(cement_papers, "水泥")
    
    bridge = detector.calculate_semantic_bridge(sig_battery, sig_cement)
    
    print(f"\n领域相似度:")
    print(f"  Jaccard: {bridge['similarity_jaccard']:.3f}")
    print(f"  余弦: {bridge['similarity_cosine']:.3f}")
    print(f"  桥梁强度: {bridge['bridge_strength']:.3f}")
    
    print(f"\n桥梁术语（Top 5）:")
    for term, weight in bridge['bridge_terms'][:5]:
        print(f"  • {term}: {weight:.3f}")


if __name__ == "__main__":
    example_usage()
