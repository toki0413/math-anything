"""
多数据源组合搜索 - 解决单一数据库覆盖不足问题

整合多个数据源：
1. OpenAlex - 开放学术图
2. CrossRef - DOI元数据
3. Semantic Scholar - AI增强学术搜索
4. CORE - 开放获取论文聚合
5. OpenAIRE - 欧盟资助项目
"""

import requests
import json
from collections import defaultdict
from datetime import datetime


def search_openalex(query, year_from=2018, year_to=2026, max_results=200):
    """OpenAlex搜索 - 扩大返回数量"""
    base_url = "https://api.openalex.org/works"
    
    all_results = []
    cursor = "*"  # 分页游标
    
    while len(all_results) < max_results and cursor:
        params = {
            "search": query,
            "filter": f"from_publication_date:{year_from}-01-01,to_publication_date:{year_to}-12-31",
            "per-page": min(200, max_results - len(all_results)),
            "cursor": cursor,
            "mailto": "user@example.com"
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            results = data.get("results", [])
            all_results.extend(results)
            
            # 获取下一页游标
            cursor = data.get("meta", {}).get("next_cursor")
            
            print(f"  OpenAlex: 已获取 {len(all_results)} 篇...")
            
            # 如果没有更多结果，退出
            if not results or not cursor:
                break
                
        except Exception as e:
            print(f"  OpenAlex错误: {e}")
            break
    
    return all_results


def search_semantic_scholar(query, year_from=2018, year_to=2026, max_results=100):
    """Semantic Scholar搜索 - 对工程领域覆盖较好"""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    all_results = []
    offset = 0
    
    while len(all_results) < max_results:
        params = {
            "query": query,
            "fields": "title,authors,year,citationCount,abstract,venue",
            "limit": min(100, max_results - len(all_results)),
            "offset": offset
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            results = data.get("data", [])
            if not results:
                break
                
            all_results.extend(results)
            offset += len(results)
            
            print(f"  Semantic Scholar: 已获取 {len(all_results)} 篇...")
            
        except Exception as e:
            print(f"  Semantic Scholar错误: {e}")
            break
    
    return all_results


def search_crossref(query, year_from=2018, year_to=2026, max_results=100):
    """CrossRef搜索 - DOI元数据"""
    base_url = "https://api.crossref.org/works"
    
    params = {
        "query": query,
        "filter": f"from-pub-date:{year_from},until-pub-date:{year_to}",
        "rows": min(1000, max_results),
        "sort": "relevance",
        "order": "desc"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        data = response.json()
        
        results = data.get("message", {}).get("items", [])
        print(f"  CrossRef: 已获取 {len(results)} 篇")
        return results
        
    except Exception as e:
        print(f"  CrossRef错误: {e}")
        return []


def search_core(query, max_results=100):
    """CORE搜索 - 开放获取论文"""
    base_url = "https://api.core.ac.uk/v3/search/works"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "limit": min(100, max_results),
        "sort": "relevance"
    }
    
    try:
        # CORE需要API key，这里尝试无key访问
        response = requests.post(base_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"  CORE: 已获取 {len(results)} 篇")
            return results
        else:
            print(f"  CORE: 需要API key (状态码 {response.status_code})")
            return []
            
    except Exception as e:
        print(f"  CORE错误: {e}")
        return []


def normalize_paper(paper, source):
    """标准化不同来源的论文数据"""
    
    if source == "openalex":
        return {
            "id": paper.get("id", "").replace("https://openalex.org/", ""),
            "title": paper.get("display_name", ""),
            "authors": [a.get("author", {}).get("display_name", "") 
                       for a in paper.get("authorships", [])],
            "year": paper.get("publication_year", 0),
            "citations": paper.get("cited_by_count", 0),
            "abstract": paper.get("abstract", ""),
            "venue": paper.get("host_venue", {}).get("display_name", ""),
            "doi": paper.get("doi", ""),
            "source": "openalex",
            "url": paper.get("id", "")
        }
    
    elif source == "semantic_scholar":
        return {
            "id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "authors": [a.get("name", "") for a in paper.get("authors", [])],
            "year": paper.get("year", 0),
            "citations": paper.get("citationCount", 0),
            "abstract": paper.get("abstract", ""),
            "venue": paper.get("venue", ""),
            "doi": "",
            "source": "semantic_scholar",
            "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
        }
    
    elif source == "crossref":
        return {
            "id": paper.get("DOI", ""),
            "title": paper.get("title", [""])[0] if isinstance(paper.get("title"), list) else paper.get("title", ""),
            "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() 
                       for a in paper.get("author", [])],
            "year": paper.get("published-print", {}).get("date-parts", [[0]])[0][0] if paper.get("published-print") else 0,
            "citations": paper.get("is-referenced-by-count", 0),
            "abstract": "",
            "venue": paper.get("container-title", [""])[0] if isinstance(paper.get("container-title"), list) else "",
            "doi": paper.get("DOI", ""),
            "source": "crossref",
            "url": f"https://doi.org/{paper.get('DOI', '')}"
        }
    
    return {}


def deduplicate_papers(papers_list):
    """去重 - 基于DOI或标题相似度"""
    
    seen_dois = set()
    seen_titles = set()
    unique_papers = []
    
    for paper in papers_list:
        # 使用DOI去重
        doi = paper.get("doi", "").lower()
        if doi and doi in seen_dois:
            continue
        
        # 使用标题去重（简化比较）
        title = paper.get("title", "").lower().replace(" ", "")[:50]
        if title and title in seen_titles:
            continue
        
        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)
        
        unique_papers.append(paper)
    
    return unique_papers


def main():
    """主函数 - 多数据源搜索"""
    
    print("\n" + "█" * 80)
    print("█" + " 多数据源组合搜索 - 水泥基电化学储能 ".center(76) + "█")
    print("█" * 80)
    
    # 查询词
    queries = [
        "cement supercapacitor",
        "concrete energy storage", 
        "conductive cement electrode",
        "carbon cement composite energy",
        "structural supercapacitor cement"
    ]
    
    all_papers = []
    
    for query in queries:
        print(f"\n查询词: '{query}'")
        print("-" * 60)
        
        # OpenAlex
        print("搜索 OpenAlex...")
        openalex_results = search_openalex(query, max_results=100)
        for r in openalex_results:
            all_papers.append(normalize_paper(r, "openalex"))
        
        # Semantic Scholar
        print("搜索 Semantic Scholar...")
        ss_results = search_semantic_scholar(query, max_results=50)
        for r in ss_results:
            all_papers.append(normalize_paper(r, "semantic_scholar"))
        
        # CrossRef
        print("搜索 CrossRef...")
        crossref_results = search_crossref(query, max_results=50)
        for r in crossref_results:
            all_papers.append(normalize_paper(r, "crossref"))
    
    print("\n" + "=" * 60)
    print(f"原始获取总数: {len(all_papers)} 篇")
    
    # 去重
    unique_papers = deduplicate_papers(all_papers)
    print(f"去重后总数: {len(unique_papers)} 篇")
    
    # 年份分布
    year_counts = defaultdict(int)
    for p in unique_papers:
        year = p.get("year", 0)
        if 2018 <= year <= 2026:
            year_counts[year] += 1
    
    print("\n年份分布（去重后）:")
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        bar = "█" * count
        print(f"  {year}: {bar} {count}")
    
    # 数据源分布
    source_counts = defaultdict(int)
    for p in unique_papers:
        source_counts[p.get("source", "unknown")] += 1
    
    print("\n数据源分布:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}篇")
    
    # 高被引论文
    top_cited = sorted(unique_papers, key=lambda x: x.get("citations", 0), reverse=True)[:10]
    
    print("\n高被引论文（Top 10）:")
    for i, p in enumerate(top_cited, 1):
        print(f"{i}. {p['title'][:60]}...")
        print(f"   引用: {p['citations']} | 年份: {p['year']} | 来源: {p['source']}")
    
    # 保存结果
    output = {
        "fetch_time": datetime.now().isoformat(),
        "queries": queries,
        "total_papers": len(unique_papers),
        "year_distribution": dict(year_counts),
        "source_distribution": dict(source_counts),
        "papers": unique_papers
    }
    
    output_file = "multi_source_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    print("\n" + "=" * 60)
    print("数据覆盖评估:")
    print("=" * 60)
    
    total = len(unique_papers)
    if total < 50:
        print("⚠️  覆盖度: 低 - 这个领域可能非常小众或新兴")
    elif total < 200:
        print("⚠️  覆盖度: 中 - 有一定研究基础但规模有限")
    else:
        print("✅ 覆盖度: 高 - 活跃的研究领域")
    
    recent_years = sum(1 for p in unique_papers if p.get("year", 0) >= 2022)
    print(f"  近3年论文占比: {recent_years/total*100:.1f}% ({recent_years}/{total})")
    
    if year_counts.get(2025, 0) + year_counts.get(2024, 0) < 5:
        print("  说明: 2024-2025年论文较少，可能是因为:")
        print("    1. 领域处于早期，论文产出有限")
        print("    2. 数据库索引延迟")
        print("    3. 研究转向相关但不同的方向")


if __name__ == "__main__":
    main()
