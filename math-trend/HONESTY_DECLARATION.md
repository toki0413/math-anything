# Math-Trend 数据真实性声明

## 当前限制（诚实说明）

### 哪些是模拟数据（Mock Data）
以下数据是**基于合理假设生成的示例数据**，用于演示分析框架：

| 数据类型 | 示例值 | 真实性 |
|---------|-------|-------|
| 论文数（328篇） | 模拟 | ❌ 虚构 |
| 引用数（1847次） | 模拟 | ❌ 虚构 |
| H指数（28） | 模拟 | ❌ 虚构 |
| 资助额（$12.5M） | 模拟 | ❌ 虚构 |
| 专利数（34项） | 模拟 | ❌ 虚构 |
| 研究者姓名 | 基于真实研究者 | ⚠️ 混合 |
| 期刊排名 | 基于领域常识 | ⚠️ 部分真实 |

### 哪些可以真实获取
通过以下API可以获取**真实数据**：

| 数据源 | 可获取数据 | 获取方式 |
|-------|-----------|---------|
| **OpenAlex** | 论文标题、作者、引用、发表时间 | API免费 |
| **CrossRef** | 论文元数据、DOI、引用关系 | API免费 |
| **USPTO** | 美国专利全文、申请人、引用 | API免费 |
| **EPO** | 欧洲专利数据 | 注册后免费 |
| **PatentsView** | 美国专利结构化数据 | API免费 |
| **Semantic Scholar** | 论文、引用、作者信息 | API免费 |
| **OpenAIRE** | 欧盟资助项目、论文 | API免费 |

---

## 如何接入真实数据

### 步骤1：获取真实论文数据（OpenAlex）

```python
import requests

def get_real_papers(query, year_from=2019, year_to=2026, max_results=200):
    """从OpenAlex获取真实论文数据"""
    
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": f"from_publication_date:{year_from}-01-01,to_publication_date:{year_to}-12-31",
        "per-page": min(max_results, 200),
        "sort": "cited_by_count:desc"
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    papers = []
    for work in data.get("results", []):
        paper = {
            "id": work.get("id"),
            "title": work.get("display_name"),
            "authors": [auth["author"]["display_name"] 
                       for auth in work.get("authorships", [])],
            "year": work.get("publication_year"),
            "citations": work.get("cited_by_count", 0),
            "concepts": [c["display_name"] for c in work.get("concepts", [])],
            "journal": work.get("host_venue", {}).get("display_name", ""),
            "doi": work.get("doi")
        }
        papers.append(paper)
    
    return papers

# 真实查询示例
papers = get_real_papers("cement supercapacitor energy storage", 
                         year_from=2020, max_results=100)
print(f"找到 {len(papers)} 篇真实论文")
print(f"示例: {papers[0]['title'][:80]}...")
print(f"引用: {papers[0]['citations']}次")
```

### 步骤2：获取真实专利数据（PatentsView）

```python
def get_real_patents(query, year_from=2020, max_results=100):
    """从PatentsView获取真实美国专利数据"""
    
    url = "https://api.patentsview.org/patents/query"
    
    query_body = {
        "q": {
            "_text_any": {"patent_title": query}
        },
        "f": ["patent_id", "patent_title", "patent_date", 
              "patent_num_cited_by_us_patents", "assignee_name"],
        "o": {"per_page": max_results}
    }
    
    response = requests.post(url, json=query_body)
    data = response.json()
    
    patents = []
    for patent in data.get("patents", []):
        p = {
            "id": patent.get("patent_id"),
            "title": patent.get("patent_title"),
            "date": patent.get("patent_date"),
            "citations": patent.get("patent_num_cited_by_us_patents", 0),
            "assignee": patent.get("assignee_name", ["Unknown"])[0] if patent.get("assignee_name") else "Unknown"
        }
        patents.append(p)
    
    return patents

# 真实查询示例
patents = get_real_patents("cement conductive energy storage", 
                          year_from=2020, max_results=50)
print(f"找到 {len(patents)} 项真实专利")
```

### 步骤3：真实数据分析示例

```python
def analyze_real_data(papers):
    """对真实数据进行分析"""
    
    # 年度分布（真实）
    year_counts = {}
    for p in papers:
        year = p.get("year", 0)
        year_counts[year] = year_counts.get(year, 0) + 1
    
    # 引用统计（真实）
    citations = [p.get("citations", 0) for p in papers]
    avg_citations = sum(citations) / len(citations) if citations else 0
    
    # 作者统计（真实）
    author_counts = {}
    for p in papers:
        for author in p.get("authors", []):
            author_counts[author] = author_counts.get(author, 0) + 1
    
    top_authors = sorted(author_counts.items(), 
                        key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_papers": len(papers),
        "year_distribution": dict(sorted(year_counts.items())),
        "avg_citations": avg_citations,
        "max_citations": max(citations) if citations else 0,
        "top_authors": top_authors
    }

# 运行真实分析
papers = get_real_papers("cement supercapacitor", max_results=100)
analysis = analyze_real_data(papers)

print("真实数据分析结果:")
print(f"  论文总数: {analysis['total_papers']}")
print(f"  平均引用: {analysis['avg_citations']:.1f}")
print(f"  最高引用: {analysis['max_citations']}")
print(f"  活跃作者: {analysis['top_authors'][:3]}")
```

---

## 改进后的诚实示例

### 标注数据来源

```python
print("╔══════════════════════════════════════════════════════════════╗")
print("║                    数据来源声明                               ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()
print("【真实数据】来自 OpenAlex API（2026-04-24查询）")
print(f"  • 查询词: 'cement supercapacitor'")
print(f"  • 时间范围: 2019-2026")
print(f"  • 获取论文数: {len(real_papers)}篇")
print()
print("【模拟数据】基于领域知识的合理估算")
print("  • 技术成熟度评估（TRL）")
print("  • 市场预测数据")
print("  • 部分性能指标")
print()
```

---

## 建议的改进方案

### 方案1: 纯真实数据版本（推荐）
- 只使用OpenAlex、PatentsView等可验证的数据
- 明确标注每个数据的来源和时间
- 不提供估算数据

### 方案2: 混合标注版本（当前）
- 真实数据 + 模拟数据混合
- **强制要求**：每个数据点标注来源（[REAL] / [ESTIMATED]）
- 提供置信度区间

### 方案3: 敏感性分析版本
- 对关键假设提供多情景分析
- 例如："如果论文年增长率为20%-40%，则..."

---

## 数据验证清单

发布报告前检查：
- [ ] 每个数字都有数据来源标注
- [ ] 模拟数据用 [SIMULATED] 标记
- [ ] 真实数据用 [REAL] 标记并注明时间
- [ ] 估算数据用 [ESTIMATED] 并给出依据
- [ ] 提供原始数据获取方法

---

## 用户自查方法

你可以通过以下方式验证数据真实性：

1. **OpenAlex**: https://openalex.org/works?page=1&filter=default.search:cement%20supercapacitor
2. **Google Scholar**: https://scholar.google.com/scholar?q=cement+supercapacitor
3. **PatentsView**: https://www.patentsview.org/search/
4. **USPTO**: https://patft.uspto.gov/

---

*本声明旨在提高透明度，帮助用户区分真实数据和模拟演示数据。*
