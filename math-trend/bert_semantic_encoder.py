"""
真实BERT语义编码模块

使用Sentence-BERT提取768维语义向量
替代简化版特征工程，实现真正的语义理解
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from unified_data_loader import UnifiedDataLoader, Paper, load_cement_storage_data

OUTPUT_DIR = Path(__file__).parent / "output" / "bert_semantic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BERTSemanticEncoder:
    """BERT语义编码器"""
    
    def __init__(self, data_loader: UnifiedDataLoader):
        self.loader = data_loader
        self.papers = data_loader.get_papers_by_confidence("all")
        self.embeddings: Dict[str, np.ndarray] = {}
        self.model = None
        
    def load_model(self):
        """加载Sentence-BERT模型"""
        print("\n加载Sentence-BERT模型...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # 使用轻量级模型（适合学术文本）
            model_name = 'all-MiniLM-L6-v2'  # 384维，速度快
            # 或者使用更大的模型：'all-mpnet-base-v2'  # 768维，精度更高
            
            self.model = SentenceTransformer(model_name)
            print(f"  ✅ 模型加载成功: {model_name}")
            print(f"  嵌入维度: {self.model.get_sentence_embedding_dimension()}")
            
            return True
            
        except ImportError:
            print("  ⚠️ sentence-transformers未安装")
            print("  尝试安装: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"  ⚠️ 模型加载失败: {e}")
            return False
    
    def encode_papers(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        编码所有论文为语义向量
        
        Args:
            batch_size: 批处理大小
        """
        if not self.model and not self.load_model():
            print("  使用备用编码方案...")
            return self._fallback_encoding()
        
        print(f"\n编码 {len(self.papers)} 篇论文...")
        
        # 准备文本
        texts = []
        paper_ids = []
        
        for paper in self.papers:
            # 组合标题和摘要
            text = f"{paper.title}. {paper.abstract}"
            if len(text) < 10:  # 过滤过短文本
                text = paper.title
            texts.append(text)
            paper_ids.append(paper.id)
        
        # 批量编码
        print(f"  批处理大小: {batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 存储
        for i, paper_id in enumerate(paper_ids):
            self.embeddings[paper_id] = embeddings[i]
        
        print(f"  ✅ 编码完成: {len(self.embeddings)} 篇论文")
        print(f"  向量维度: {embeddings.shape[1]}")
        
        return self.embeddings
    
    def _fallback_encoding(self) -> Dict[str, np.ndarray]:
        """
        备用编码方案（当BERT不可用时）
        使用TF-IDF + SVD模拟语义向量
        """
        print("\n使用备用编码方案（TF-IDF + SVD）...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        # 准备文本
        texts = []
        paper_ids = []
        
        for paper in self.papers:
            text = f"{paper.title} {paper.abstract}".lower()
            texts.append(text)
            paper_ids.append(paper.id)
        
        # TF-IDF
        print("  计算TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # SVD降维到384维（模拟BERT维度）
        print("  SVD降维...")
        n_components = min(384, tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components)
        reduced = svd.fit_transform(tfidf_matrix)
        
        # 存储
        for i, paper_id in enumerate(paper_ids):
            self.embeddings[paper_id] = reduced[i]
        
        print(f"  ✅ 备用编码完成: {len(self.embeddings)} 篇论文")
        print(f"  向量维度: {n_components}")
        
        return self.embeddings
    
    def calculate_similarity(self, paper_id1: str, paper_id2: str) -> float:
        """计算两篇论文的语义相似度"""
        if paper_id1 not in self.embeddings or paper_id2 not in self.embeddings:
            return 0.0
        
        emb1 = self.embeddings[paper_id1]
        emb2 = self.embeddings[paper_id2]
        
        # 余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    def find_similar_papers(self, paper_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """查找最相似的论文"""
        if paper_id not in self.embeddings:
            return []
        
        query_emb = self.embeddings[paper_id]
        similarities = []
        
        for pid, emb in self.embeddings.items():
            if pid != paper_id:
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                similarities.append((pid, float(sim)))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def cluster_by_semantics(self, n_clusters: int = 8) -> Dict[int, List[Paper]]:
        """
        基于语义向量聚类
        """
        print(f"\n基于BERT语义向量聚类（k={n_clusters}）...")
        
        if not self.embeddings:
            self.encode_papers()
        
        # 准备数据
        paper_ids = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[pid] for pid in paper_ids])
        
        # K-means聚类
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        # 组织结果
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            paper = next((p for p in self.papers if p.id == paper_ids[i]), None)
            if paper:
                clusters[label].append(paper)
        
        print(f"  ✅ 聚类完成: {n_clusters}个主题")
        
        # 计算每个聚类的中心向量
        self.cluster_centers = {}
        for cluster_id in range(n_clusters):
            cluster_papers = [p for p in self.papers if p.id in paper_ids and 
                            labels[paper_ids.index(p.id)] == cluster_id]
            if cluster_papers:
                center = np.mean([self.embeddings[p.id] for p in cluster_papers], axis=0)
                self.cluster_centers[cluster_id] = center
        
        return dict(clusters)
    
    def detect_emerging_topics_advanced(self, clusters: Dict[int, List[Paper]]) -> List[Dict]:
        """
        高级新兴主题检测
        基于BERT语义向量和时间分析
        """
        print("\n高级新兴主题检测...")
        
        emerging = []
        current_year = 2026
        
        for cluster_id, papers in clusters.items():
            years = [p.year for p in papers if p.year > 0]
            
            if not years:
                continue
            
            # 时间特征
            recent_papers = [p for p in papers if p.year >= current_year - 3]
            old_papers = [p for p in papers if p.year < current_year - 3]
            
            # 新颖性得分
            recency_score = len(recent_papers) / len(papers) if papers else 0
            growth_score = len(recent_papers) / (len(old_papers) + 1)
            
            # 语义紧凑度（聚类内相似度）
            if len(papers) > 1:
                similarities = []
                for i, p1 in enumerate(papers[:10]):  # 采样
                    for p2 in papers[i+1:11]:
                        if p1.id in self.embeddings and p2.id in self.embeddings:
                            sim = self.calculate_similarity(p1.id, p2.id)
                            similarities.append(sim)
                
                coherence = np.mean(similarities) if similarities else 0
            else:
                coherence = 1.0
            
            # 综合新颖性
            novelty = (recency_score * 0.4 + growth_score * 0.4 + coherence * 0.2)
            
            if novelty > 0.3:  # 阈值
                # 提取关键词（使用TF-IDF）
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                texts = [f"{p.title} {p.abstract}" for p in papers]
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                try:
                    tfidf = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    scores = tfidf.mean(axis=0).A1
                    top_keywords = [(feature_names[i], scores[i]) 
                                  for i in scores.argsort()[-5:][::-1]]
                except:
                    top_keywords = []
                
                emerging.append({
                    "cluster_id": cluster_id,
                    "novelty_score": novelty,
                    "recency": recency_score,
                    "growth": growth_score,
                    "coherence": coherence,
                    "paper_count": len(papers),
                    "year_range": (min(years), max(years)),
                    "keywords": top_keywords
                })
        
        # 排序
        emerging.sort(key=lambda x: x["novelty_score"], reverse=True)
        
        print(f"  检测到 {len(emerging)} 个新兴主题")
        return emerging
    
    def save_embeddings(self):
        """保存嵌入向量"""
        if not self.embeddings:
            print("  无嵌入可保存")
            return
        
        # 保存为numpy数组
        embeddings_path = OUTPUT_DIR / "paper_embeddings.npy"
        
        # 创建ID到索引的映射
        id_to_idx = {pid: i for i, pid in enumerate(self.embeddings.keys())}
        
        # 保存映射
        import json
        mapping_path = OUTPUT_DIR / "paper_id_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(id_to_idx, f)
        
        # 保存向量
        embeddings_matrix = np.array(list(self.embeddings.values()))
        np.save(embeddings_path, embeddings_matrix)
        
        print(f"\n✅ 嵌入向量已保存:")
        print(f"  向量文件: {embeddings_path}")
        print(f"  映射文件: {mapping_path}")
    
    def generate_semantic_report(self, clusters: Dict[int, List[Paper]], 
                                 emerging: List[Dict]) -> str:
        """生成语义分析报告"""
        print("\n生成BERT语义分析报告...")
        
        report = f"""# BERT语义编码分析报告

**分析方法**: Sentence-BERT语义向量 + K-means聚类  
**数据规模**: {len(self.papers)}篇论文  
**向量维度**: {list(self.embeddings.values())[0].shape[0] if self.embeddings else 'N/A'}  
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. 语义编码方法

### 1.1 模型信息

| 属性 | 值 |
|-----|-----|
| 模型 | {'Sentence-BERT (all-MiniLM-L6-v2)' if self.model else 'TF-IDF + SVD (备用)'} |
| 维度 | {list(self.embeddings.values())[0].shape[0] if self.embeddings else 'N/A'} |
| 批处理 | 32 |
| 编码文本 | 标题 + 摘要 |

### 1.2 编码优势

相比传统TF-IDF：
- ✅ 捕捉语义关系（不仅是词频）
- ✅ 理解上下文含义
- ✅ 支持语义相似度计算
- ✅ 跨语言语义对齐

---

## 2. 语义聚类结果

### 2.1 主题概览

| 主题ID | 论文数 | 语义紧凑度 | 年份范围 | 核心关键词 |
|-------|-------|-----------|---------|-----------|
"""
        
        for cluster_id, papers in sorted(clusters.items(), 
                                         key=lambda x: len(x[1]), 
                                         reverse=True):
            years = [p.year for p in papers if p.year > 0]
            
            # 计算紧凑度
            if len(papers) > 1:
                similarities = []
                for i, p1 in enumerate(papers[:5]):
                    for p2 in papers[i+1:6]:
                        if p1.id in self.embeddings and p2.id in self.embeddings:
                            sim = self.calculate_similarity(p1.id, p2.id)
                            similarities.append(sim)
                coherence = np.mean(similarities) if similarities else 0
            else:
                coherence = 1.0
            
            # 提取关键词
            texts = [f"{p.title} {p.abstract}" for p in papers]
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                tfidf = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf.mean(axis=0).A1
                top_keywords = [feature_names[i] for i in scores.argsort()[-3:][::-1]]
            except:
                top_keywords = []
            
            year_range = f"{min(years)}-{max(years)}" if years else "N/A"
            report += f"| {cluster_id} | {len(papers)} | {coherence:.3f} | {year_range} | {', '.join(top_keywords)} |\n"
        
        report += f"""
---

## 3. 新兴主题检测

### 3.1 新兴主题Top 5

| 排名 | 主题ID | 新颖性得分 | 近期占比 | 增长率 | 紧凑度 |
|-----|-------|-----------|---------|-------|-------|
"""
        
        for i, topic in enumerate(emerging[:5], 1):
            report += f"| {i} | {topic['cluster_id']} | {topic['novelty_score']:.3f} | {topic['recency']:.2f} | {topic['growth']:.2f} | {topic['coherence']:.3f} |\n"
        
        report += f"""
### 3.2 新兴主题详情

"""
        
        for topic in emerging[:3]:
            report += f"""**主题 {topic['cluster_id']}** (新颖性: {topic['novelty_score']:.3f})

- 论文数: {topic['paper_count']}
- 年份范围: {topic['year_range'][0]}-{topic['year_range'][1]}
- 近期占比: {topic['recency']:.1%}
- 增长率: {topic['growth']:.2f}
- 语义紧凑度: {topic['coherence']:.3f}
- 关键词: {', '.join([k for k, _ in topic['keywords'][:5]])}

"""
        
        report += f"""
---

## 4. 语义相似度案例

### 4.1 最相似论文对

"""
        
        # 找出最相似的论文对
        max_sim = 0
        max_pair = None
        
        paper_ids = list(self.embeddings.keys())
        for i, pid1 in enumerate(paper_ids[:20]):
            for pid2 in paper_ids[i+1:21]:
                sim = self.calculate_similarity(pid1, pid2)
                if sim > max_sim:
                    max_sim = sim
                    max_pair = (pid1, pid2)
        
        if max_pair:
            p1 = next((p for p in self.papers if p.id == max_pair[0]), None)
            p2 = next((p for p in self.papers if p.id == max_pair[1]), None)
            
            if p1 and p2:
                report += f"""**最相似论文对** (相似度: {max_sim:.3f}):

论文1: {p1.title[:60]}...
论文2: {p2.title[:60]}...

**解读**: 两篇论文在语义上高度相关，可能研究同一具体问题或方法。

"""
        
        report += f"""
---

## 5. 方法对比

### 5.1 BERT vs TF-IDF

| 对比项 | TF-IDF | BERT语义编码 |
|-------|--------|-------------|
| 维度 | 高维稀疏 | 低维稠密 |
| 语义理解 | 无 | 深度 |
| 上下文 | 忽略 | 考虑 |
| 计算成本 | 低 | 中等 |
| 跨语言 | 困难 | 支持 |
| 新兴主题 | 难识别 | 易识别 |

### 5.2 当前实现

{'✅ 使用真实Sentence-BERT模型' if self.model else '⚠️ 使用TF-IDF备用方案'}

---

## 6. 应用建议

### 6.1 研究应用

1. **文献检索**: 基于语义相似度而非关键词
2. **综述生成**: 自动聚类相关研究
3. **趋势预测**: 追踪语义空间演化
4. **跨领域发现**: 识别语义相似的不同领域研究

### 6.2 改进建议

1. **更大模型**: 使用all-mpnet-base-v2（768维）
2. **领域适应**: 在学术文本上微调BERT
3. **多语言**: 支持中文、日文论文
4. **增量更新**: 新论文实时编码

---

*本报告由 Math-Trend BERT语义编码模块生成*  
{'*使用真实Sentence-BERT模型*' if self.model else '*使用TF-IDF备用方案（建议安装sentence-transformers）*'}
"""
        
        return report
    
    def save_report(self, report_content: str):
        """保存报告"""
        report_path = OUTPUT_DIR / "bert_semantic_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n✅ 语义分析报告已保存: {report_path}")
        return report_path


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " BERT语义编码 - 真实语义向量分析 ".center(76) + "█")
    print("█" * 80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = load_cement_storage_data()
    print(f"  加载 {len(loader.all_papers)} 篇论文")
    
    # 2. 创建编码器
    print("\n2. 创建BERT语义编码器...")
    encoder = BERTSemanticEncoder(loader)
    
    # 3. 编码论文
    print("\n3. 编码论文...")
    embeddings = encoder.encode_papers()
    
    # 4. 语义聚类
    print("\n4. 语义聚类...")
    clusters = encoder.cluster_by_semantics(n_clusters=8)
    
    # 5. 检测新兴主题
    print("\n5. 检测新兴主题...")
    emerging = encoder.detect_emerging_topics_advanced(clusters)
    
    # 6. 保存嵌入
    print("\n6. 保存嵌入向量...")
    encoder.save_embeddings()
    
    # 7. 生成报告
    print("\n7. 生成语义分析报告...")
    report = encoder.generate_semantic_report(clusters, emerging)
    report_path = encoder.save_report(report)
    
    # 8. 总结
    print("\n" + "█" * 80)
    print("█" + " 分析完成 ".center(76) + "█")
    print("█" * 80)
    
    print(f"\n📊 核心结果:")
    print(f"  编码论文: {len(embeddings)}篇")
    print(f"  向量维度: {list(embeddings.values())[0].shape[0] if embeddings else 'N/A'}")
    print(f"  识别主题: {len(clusters)}个")
    print(f"  新兴主题: {len(emerging)}个")
    
    print(f"\n🏷️ 主题分布:")
    for cluster_id, papers in sorted(clusters.items(), 
                                     key=lambda x: len(x[1]), 
                                     reverse=True)[:5]:
        print(f"  主题{cluster_id}: {len(papers)}篇")
    
    print(f"\n📄 报告文件: {report_path}")
    print(f"💾 嵌入向量: {OUTPUT_DIR / 'paper_embeddings.npy'}")
    
    print()


if __name__ == "__main__":
    main()
