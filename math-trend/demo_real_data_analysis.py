"""
水泥基电化学储能 - 仅使用真实数据的分析

数据来源：
- OpenAlex API (论文数据) - 真实、可验证
- PatentsView API (美国专利) - 真实、可验证

所有数据均可通过提供的链接或API调用验证
"""

import requests
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent / "output" / "real_data_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_openalex_papers(query, year_from=2018, year_to=2026, max_results=100):
    """
    从OpenAlex获取真实论文数据
    
    API文档: https://docs.openalex.org/api-entities/works/search-works
    数据验证: https://openalex.org/works?page=1&filter=default.search:{query}
    """
    base_url = "https://api.openalex.org/works"
    
    params = {
        "search": query,
        "filter": f"from_publication_date:{year_from}-01-01,to_publication_date:{year_to}-12-31",
        "per-page": min(max_results, 200),
        "sort": "cited_by_count:desc",
        "mailto": "user@example.com"  # 有礼貌地使用API
    }
    
    print(f"正在查询 OpenAlex API: '{query}'")
    print(f"URL: {base_url}?search={query.replace(' ', '%20')}")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        papers = []
        for work in data.get("results", []):
            paper = {
                "id": work.get("id", "").replace("https://openalex.org/", ""),
                "title": work.get("display_name", "N/A"),
                "authors": [
                    auth.get("author", {}).get("display_name", "Unknown")
                    for auth in work.get("authorships", [])
                ],
                "year": work.get("publication_year", 0),
                "citations": work.get("cited_by_count", 0),
                "concepts": [
                    c.get("display_name", "") 
                    for c in work.get("concepts", [])
                ],
                "journal": work.get("primary_location", {}).get("source", {}).get("display_name", "Unknown"),
                "doi": work.get("doi", "N/A"),
                "openalex_url": work.get("id", "")
            }
            papers.append(paper)
        
        print(f"✅ 成功获取 {len(papers)} 篇真实论文")
        return papers
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API请求失败: {e}")
        print("提示: 检查网络连接或稍后再试")
        return []


def fetch_patentsview_patents(query, year_from=2018, max_results=100):
    """
    从PatentsView获取真实美国专利数据
    
    API文档: https://api.patentsview.org/doc.html
    数据验证: https://www.patentsview.org/search/
    """
    url = "https://api.patentsview.org/patents/query"
    
    # 构建查询 - 在标题或摘要中搜索
    query_body = {
        "q": {
            "_or": [
                {"_text_any": {"patent_title": query}},
                {"_text_any": {"patent_abstract": query}}
            ]
        },
        "f": [
            "patent_id",
            "patent_title", 
            "patent_date",
            "patent_num_cited_by_us_patents",
            "assignee_organization"
        ],
        "o": {
            "per_page": max_results,
            "matched_subentities_only": True
        }
    }
    
    print(f"\n正在查询 PatentsView API: '{query}'")
    
    try:
        response = requests.post(url, json=query_body, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        patents = []
        for patent in data.get("patents", []):
            # 过滤年份
            patent_date = patent.get("patent_date", "1900-01-01")
            year = int(patent_date.split("-")[0]) if patent_date else 1900
            
            if year_from <= year <= 2026:
                p = {
                    "id": patent.get("patent_id", "N/A"),
                    "title": patent.get("patent_title", "N/A"),
                    "date": patent_date,
                    "year": year,
                    "citations": patent.get("patent_num_cited_by_us_patents", 0),
                    "assignee": patent.get("assignee_organization", ["Unknown"])[0] if patent.get("assignee_organization") else "Unknown"
                }
                patents.append(p)
        
        print(f"✅ 成功获取 {len(patents)} 项真实美国专利")
        return patents
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API请求失败: {e}")
        return []


def analyze_papers_real(papers):
    """对真实论文数据进行分析"""
    
    if not papers:
        return {"error": "No papers fetched"}
    
    # 年度分布
    year_counts = defaultdict(int)
    for p in papers:
        year = p.get("year", 0)
        if year > 0:
            year_counts[year] += 1
    
    # 引用统计
    citations = [p.get("citations", 0) for p in papers]
    total_citations = sum(citations)
    avg_citations = total_citations / len(citations) if citations else 0
    max_citations = max(citations) if citations else 0
    
    # 作者统计
    author_counts = defaultdict(int)
    for p in papers:
        for author in p.get("authors", [])[:5]:  # 只统计前5作者
            if author and author != "Unknown":
                author_counts[author] += 1
    
    top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 期刊统计
    journal_counts = defaultdict(int)
    for p in papers:
        journal = p.get("journal", "Unknown")
        if journal and journal != "Unknown":
            journal_counts[journal] += 1
    
    top_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 高被引论文
    top_cited = sorted(papers, key=lambda x: x.get("citations", 0), reverse=True)[:5]
    
    return {
        "total_papers": len(papers),
        "total_citations": total_citations,
        "avg_citations": round(avg_citations, 1),
        "max_citations": max_citations,
        "year_range": f"{min(year_counts.keys())}-{max(year_counts.keys())}" if year_counts else "N/A",
        "year_distribution": dict(sorted(year_counts.items())),
        "top_authors": top_authors,
        "top_journals": top_journals,
        "top_cited_papers": [
            {
                "title": p["title"][:80] + "..." if len(p["title"]) > 80 else p["title"],
                "citations": p["citations"],
                "year": p["year"],
                "url": p["openalex_url"]
            }
            for p in top_cited
        ]
    }


def analyze_patents_real(patents):
    """对真实专利数据进行分析"""
    
    if not patents:
        return {"error": "No patents fetched"}
    
    # 年度分布
    year_counts = defaultdict(int)
    for p in patents:
        year = p.get("year", 0)
        if year > 0:
            year_counts[year] += 1
    
    # 引用统计
    citations = [p.get("citations", 0) for p in patents]
    total_citations = sum(citations)
    avg_citations = total_citations / len(citations) if citations else 0
    
    # 申请人统计
    assignee_counts = defaultdict(int)
    for p in patents:
        assignee = p.get("assignee", "Unknown")
        if assignee and assignee != "Unknown":
            assignee_counts[assignee] += 1
    
    top_assignees = sorted(assignee_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_patents": len(patents),
        "total_citations": total_citations,
        "avg_citations": round(avg_citations, 1),
        "year_distribution": dict(sorted(year_counts.items())),
        "top_assignees": top_assignees
    }


def print_data_sources():
    """打印数据来源声明"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " 数据来源声明 (Data Source Declaration) ".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print("本分析使用的全部数据均来自公开可验证的API：\n")
    
    print("【OpenAlex】")
    print("  • 类型: 学术论文数据")
    print("  • 来源: https://openalex.org")
    print("  • 费用: 免费")
    print("  • 验证: 可通过提供的URL直接访问查看")
    
    print("\n【PatentsView】")
    print("  • 类型: 美国专利数据")
    print("  • 来源: https://www.patentsview.org")
    print("  • 费用: 免费")
    print("  • 验证: 可通过专利号在USPTO官网验证")
    
    print("\n数据获取时间:", datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("所有数据均可通过上述链接独立验证\n")


def save_raw_data(papers, patents, query):
    """保存原始数据供验证"""
    
    # 保存论文数据
    papers_file = OUTPUT_DIR / "raw_papers.json"
    with open(papers_file, 'w', encoding='utf-8') as f:
        json.dump({
            "query": query,
            "fetch_time": datetime.now().isoformat(),
            "source": "OpenAlex API",
            "total": len(papers),
            "papers": papers
        }, f, indent=2, ensure_ascii=False)
    
    # 保存专利数据
    patents_file = OUTPUT_DIR / "raw_patents.json"
    with open(patents_file, 'w', encoding='utf-8') as f:
        json.dump({
            "query": query,
            "fetch_time": datetime.now().isoformat(),
            "source": "PatentsView API",
            "total": len(patents),
            "patents": patents
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n原始数据已保存（供验证）:")
    print(f"  • {papers_file}")
    print(f"  • {patents_file}")


def generate_real_report(paper_analysis, patent_analysis, query):
    """生成基于真实数据的报告"""
    
    report_path = OUTPUT_DIR / "real_data_report.md"
    
    report = f"""# 水泥基电化学储能领域分析报告

**数据来源**: OpenAlex API + PatentsView API  
**查询词**: "{query}"  
**数据获取时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**验证方式**: 所有数据均可通过提供的URL或专利号独立验证

---

## 1. 论文分析 (OpenAlex)

### 1.1 基本统计

| 指标 | 数值 |
|-----|------|
| 总论文数 | {paper_analysis.get('total_papers', 'N/A')} |
| 总被引次数 | {paper_analysis.get('total_citations', 'N/A')} |
| 平均被引 | {paper_analysis.get('avg_citations', 'N/A')} |
| 最高被引 | {paper_analysis.get('max_citations', 'N/A')} |
| 时间范围 | {paper_analysis.get('year_range', 'N/A')} |

### 1.2 年度分布

| 年份 | 论文数 |
|-----|-------|
"""
    
    # 添加年度分布
    year_dist = paper_analysis.get('year_distribution', {})
    for year, count in sorted(year_dist.items()):
        report += f"| {year} | {count} |\n"
    
    report += f"""
### 1.3 高被引论文 (Top 5)

"""
    
    top_papers = paper_analysis.get('top_cited_papers', [])
    for i, paper in enumerate(top_papers, 1):
        report += f"{i}. **{paper['title']}**\n"
        report += f"   - 被引: {paper['citations']}次 | 发表: {paper['year']}\n"
        report += f"   - 验证: [{paper['url']}]({paper['url']})\n\n"
    
    report += f"""
### 1.4 活跃作者 (Top 10)

| 排名 | 作者 | 论文数 |
|-----|-----|-------|
"""
    
    top_authors = paper_analysis.get('top_authors', [])
    for i, (author, count) in enumerate(top_authors, 1):
        report += f"| {i} | {author} | {count} |\n"
    
    report += f"""
### 1.5 主要期刊 (Top 10)

| 排名 | 期刊 | 论文数 |
|-----|-----|-------|
"""
    
    top_journals = paper_analysis.get('top_journals', [])
    for i, (journal, count) in enumerate(top_journals, 1):
        report += f"| {i} | {journal} | {count} |\n"
    
    # 专利部分
    report += f"""

## 2. 专利分析 (PatentsView - 美国)

### 2.1 基本统计

| 指标 | 数值 |
|-----|------|
| 总专利数 | {patent_analysis.get('total_patents', 'N/A')} |
| 总被引次数 | {patent_analysis.get('total_citations', 'N/A')} |
| 平均被引 | {patent_analysis.get('avg_citations', 'N/A')} |

### 2.2 年度分布

| 年份 | 专利数 |
|-----|-------|
"""
    
    patent_year_dist = patent_analysis.get('year_distribution', {})
    for year, count in sorted(patent_year_dist.items()):
        report += f"| {year} | {count} |\n"
    
    report += f"""
### 2.3 主要申请人 (Top 10)

| 排名 | 申请人 | 专利数 |
|-----|-------|-------|
"""
    
    top_assignees = patent_analysis.get('top_assignees', [])
    for i, (assignee, count) in enumerate(top_assignees, 1):
        report += f"| {i} | {assignee} | {count} |\n"
    
    report += f"""

## 3. 数据来源验证

### 3.1 论文验证
- OpenAlex搜索链接: https://openalex.org/works?page=1&filter=default.search:{query.replace(' ', '%20')}
- 原始数据: `raw_papers.json`

### 3.2 专利验证
- PatentsView搜索: https://www.patentsview.org/search/
- USPTO验证: https://patft.uspto.gov/
- 原始数据: `raw_patents.json`

---

**声明**: 本报告所有数据均来自公开API，可在获取时间点后通过上述链接验证。

**限制说明**:
- OpenAlex覆盖范围: 约90%的学术文献，可能存在遗漏
- PatentsView: 仅美国专利，不包含其他国家
- 引用统计: 可能有1-3个月的延迟

---

*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


def main():
    """主函数 - 仅使用真实数据"""
    
    print("\n" + "█" * 80)
    print("█" + " 水泥基电化学储能 - 仅使用真实数据分析 ".center(76) + "█")
    print("█" + " 数据来源: OpenAlex + PatentsView (全部可验证) ".center(76) + "█")
    print("█" * 80)
    
    # 打印数据来源声明
    print_data_sources()
    
    # 查询词
    query = "cement supercapacitor energy storage"
    print(f"查询词: '{query}'\n")
    
    # 获取真实论文数据
    print("=" * 80)
    papers = fetch_openalex_papers(query, year_from=2018, year_to=2026, max_results=100)
    
    if papers:
        print(f"\n示例论文（真实数据）:")
        for i, p in enumerate(papers[:3], 1):
            print(f"{i}. {p['title'][:70]}...")
            print(f"   作者: {', '.join(p['authors'][:3])}")
            print(f"   引用: {p['citations']}次 | 发表: {p['year']}")
            print(f"   验证: {p['openalex_url']}")
            print()
    
    # 获取真实专利数据
    print("=" * 80)
    patents = fetch_patentsview_patents(query, year_from=2018, max_results=100)
    
    if patents:
        print(f"\n示例专利（真实数据）:")
        for i, p in enumerate(patents[:3], 1):
            print(f"{i}. {p['title'][:70]}...")
            print(f"   申请人: {p['assignee']}")
            print(f"   引用: {p['citations']}次 | 申请: {p['year']}")
            print(f"   验证: https://patents.google.com/patent/US{p['id']}")
            print()
    
    # 分析真实数据
    print("=" * 80)
    print("【分析真实数据】")
    print("=" * 80)
    
    paper_analysis = analyze_papers_real(papers)
    patent_analysis = analyze_patents_real(patents)
    
    # 打印分析结果
    print("\n【论文分析结果】")
    print(f"  总论文数: {paper_analysis.get('total_papers', 0)}")
    print(f"  总被引: {paper_analysis.get('total_citations', 0)}")
    print(f"  平均被引: {paper_analysis.get('avg_citations', 0)}")
    print(f"  最高被引: {paper_analysis.get('max_citations', 0)}")
    
    if paper_analysis.get('year_distribution'):
        print(f"\n  年度分布:")
        for year, count in sorted(paper_analysis['year_distribution'].items()):
            bar = "█" * count
            print(f"    {year}: {bar} {count}")
    
    print("\n【专利分析结果】")
    print(f"  总专利数: {patent_analysis.get('total_patents', 0)}")
    print(f"  总被引: {patent_analysis.get('total_citations', 0)}")
    print(f"  平均被引: {patent_analysis.get('avg_citations', 0)}")
    
    # 保存原始数据
    print("\n" + "=" * 80)
    save_raw_data(papers, patents, query)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("【生成真实数据报告】")
    print("=" * 80)
    
    report_path = generate_real_report(paper_analysis, patent_analysis, query)
    print(f"\n✅ 报告已保存: {report_path}")
    
    # 总结
    print("\n" + "█" * 80)
    print("█" + " 分析完成 (仅使用真实数据) ".center(76) + "█")
    print("█" * 80)
    
    print("\n✅ 本分析保证:")
    print("  • 所有论文数据来自 OpenAlex API（可验证）")
    print("  • 所有专利数据来自 PatentsView API（可验证）")
    print("  • 原始数据已保存为JSON供核查")
    print("  • 报告中包含所有验证链接")
    
    print("\n⚠️ 数据限制:")
    print("  • OpenAlex覆盖率约90%，可能有遗漏")
    print("  • PatentsView仅包含美国专利")
    print("  • 引用统计可能有1-3个月延迟")
    
    print("\n💡 验证方法:")
    print("  1. 打开报告中的OpenAlex链接查看原始论文")
    print("  2. 使用专利号在Google Patents验证")
    print("  3. 查看保存的JSON文件获取完整原始数据")
    
    print()


if __name__ == "__main__":
    main()
