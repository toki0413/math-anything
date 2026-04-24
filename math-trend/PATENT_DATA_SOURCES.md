# 专利数据源开源接入方案

本文档汇总 Math-Trend 可用的专利数据源及其接入方式。

---

## 1. 完全开源免费

### 1.1 Google Patents Public Datasets
**网址**: https://patents.google.com/  
**数据范围**: 全球100+国家/地区，1亿+专利  
**更新频率**: 每周  
**费用**: 完全免费

**接入方式**:
```python
# 方法1: BigQuery（推荐）
from google.cloud import bigquery

client = bigquery.Client()
query = """
SELECT * 
FROM `patents-public-data.patents.publications` 
WHERE EXISTS (
  SELECT 1 FROM UNNEST(title_localized) 
  WHERE text LIKE '%cement%' AND language_code = 'en'
)
AND filing_date >= '2019-01-01'
"""
results = client.query(query)

# 方法2: RSS Feed（轻量级）
import feedparser
feed = feedparser.parse("https://patents.google.com/rss/assignee/Google")
```

**优点**:
- 数据最全，覆盖全球
- 标准化程度高
- 支持全文检索

**限制**:
- BigQuery需要Google Cloud账号（有免费额度）
- 高频查询可能产生费用

---

### 1.2 USPTO Open Data API
**网址**: https://developer.uspto.gov/api-catalog  
**数据范围**: 美国专利全文，1976年至今  
**费用**: 完全免费

**接入方式**:
```python
import requests

# 专利搜索API
url = "https://developer.uspto.gov/ibd-api/v1/application/publications"
params = {
    "searchText": "cement energy storage",
    "fq": "appFilingDate:[2019-01-01 TO 2026-12-31]",
    "rows": 100
}
response = requests.get(url, params=params)
data = response.json()

# 批量下载（Bulk Downloads）
# https://bulkdata.uspto.gov/
```

**优点**:
- 官方数据源，最权威
- 全文XML格式，结构化好
- 完全免费无限制

**限制**:
- 仅美国专利
- API速率限制（每秒1-2请求）

---

### 1.3 EPO Open Patent Services (OPS)
**网址**: https://developers.epo.org/  
**数据范围**: 欧洲专利，全球专利家族  
**费用**: 免费（需注册，有查询限制）

**接入方式**:
```python
import requests
from requests.auth import HTTPBasicAuth

# 注册获取Consumer Key和Secret
consumer_key = "your_key"
consumer_secret = "your_secret"

# 获取access token
auth_url = "https://ops.epo.org/3.2/auth/accesstoken"
response = requests.post(
    auth_url,
    auth=HTTPBasicAuth(consumer_key, consumer_secret),
    data={"grant_type": "client_credentials"}
)
token = response.json()["access_token"]

# 搜索专利
search_url = "https://ops.epo.org/3.2/rest-services/published-data/search"
headers = {"Authorization": f"Bearer {token}"}
params = {
    "q": "cement supercapacitor",
    "Range": "1-100"
}
response = requests.get(search_url, headers=headers, params=params)
```

**优点**:
- 覆盖欧洲及全球专利家族
- 专利家族关联清晰
- 法律状态信息完整

**限制**:
- 每周限制4000请求
- 需要注册申请

---

### 1.4 中国专利公布公告数据
**网址**: https://cpquery.cnipa.gov.cn/  
**数据范围**: 中国发明专利、实用新型  
**费用**: 免费

**接入方式**:
```python
# 方法1: 官方网站爬取（需遵守robots.txt）
# 方法2: 使用第三方聚合API

import requests

# 使用知轮API（示例，需申请key）
url = "http://api.zhuanlil.com/search"
params = {
    "apiKey": "your_key",
    "q": "水泥 储能",
    "type": "invention",
    "page": 1,
    "size": 100
}
response = requests.get(url, params=params)
```

**批量数据下载**:
- 国家知识产权局提供批量数据下载服务
- 需要申请并签署协议

---

## 2. 学术/研究用途免费

### 2.1 Lens.org
**网址**: https://www.lens.org/  
**数据范围**: 全球专利，学术文献关联  
**费用**: 免费（学术研究）/ 付费（商业）

**接入方式**:
```python
import requests

# Lens API
url = "https://api.lens.org/patent/search"
headers = {"Authorization": "Bearer your_access_token"}
payload = {
    "query": {
        "match_phrase": {
            "title": "cement energy storage"
        }
    },
    "size": 100
}
response = requests.post(url, headers=headers, json=payload)
```

**特点**:
- 专利与论文关联分析
- 可视化分析工具强大
- 开放学术数据

---

### 2.2 PatentsView
**网址**: https://www.patentsview.org/web/  
**数据范围**: 美国专利，1976年至今  
**费用**: 完全免费

**接入方式**:
```python
import requests

# PatentsView API
url = "https://api.patentsview.org/patents/query"
query = {
    "q": {"_text_any": {"patent_title": "cement supercapacitor"}},
    "f": ["patent_id", "patent_title", "patent_date", "assignee_name"],
    "o": {"per_page": 100}
}
response = requests.post(url, json=query)
data = response.json()
```

**特点**:
- 数据已清洗，结构化好
- 提供预构建的数据表
- 支持复杂查询

---

## 3. 推荐接入方案

### 方案1: 全球覆盖（推荐）
```python
class GlobalPatentAggregator:
    """聚合多个数据源的专利数据"""
    
    def __init__(self):
        self.sources = {
            'google_bigquery': GooglePatentsClient(),
            'uspto': USPTOClient(),
            'epo': EPOClient()
        }
    
    def search_global(self, query, year_from=2015, year_to=2026):
        """搜索全球专利"""
        all_patents = []
        
        # 并行查询多个源
        for source_name, client in self.sources.items():
            try:
                patents = client.search(query, year_from, year_to)
                for p in patents:
                    p['data_source'] = source_name
                all_patents.extend(patents)
            except Exception as e:
                print(f"{source_name}查询失败: {e}")
        
        # 去重（基于专利家族）
        return self._deduplicate(all_patents)
```

### 方案2: 轻量级（免费无账号）
```python
class LightweightPatentSearch:
    """无需API key的轻量级方案"""
    
    def __init__(self):
        self.base_urls = {
            'google': 'https://patents.google.com/?q=',
            'uspto': 'https://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&p=1&u=%2Fnetahtml%2FPTO%2Fsearch-bool.html&r=0&f=S&l=50&TERM1=',
        }
    
    def search_simple(self, query):
        """简单的网页搜索"""
        # 使用requests + BeautifulSoup爬取
        # 注意遵守robots.txt和速率限制
        pass
```

---

## 4. 数据字段映射

| 字段 | Google Patents | USPTO | EPO | Lens |
|-----|---------------|-------|-----|------|
| 专利号 | publication_number | patentNumber | doc_number | patent_id |
| 标题 | title | inventionTitle | invention_title | title |
| 摘要 | abstract | abstract | abstract | abstract |
| 申请人 | assignee | assigneeName | applicants | assignee_name |
| 申请日 | filing_date | filingDate | earliest_filing_date | filing_date |
| 公开日 | publication_date | publicationDate | publication_date | publication_date |
| 引用数 | num_citations | citationCount | npl_citation | cited_by_count |
| 分类号 | cpc/classification_cpc | classificationText | classification_ipcr | classification |

---

## 5. 实施建议

### 阶段1: 快速启动（1-2天）
- 使用 PatentsView API（美国数据，无需注册）
- 或使用 Google Patents RSS（轻量级）

### 阶段2: 扩展覆盖（1周）
- 申请 EPO OPS 账号
- 接入 BigQuery（如有Google Cloud账号）

### 阶段3: 全面覆盖（2-4周）
- 开发多源聚合器
- 实现数据去重和标准化
- 建立定期同步机制

---

## 6. 注意事项

### 法律合规
- 遵守各数据源的使用条款
- 遵守 robots.txt
- 注意API速率限制
- 学术使用需注明数据来源

### 数据质量
- 不同源的数据格式需标准化
- 专利家族需要去重
- 时区、日期格式需统一
- 申请人名称需归一化（公司名变体）

### 成本估算
| 方案 | 月查询量 | 预估成本 |
|-----|---------|---------|
| PatentsView | 10,000 | $0 |
| EPO OPS | 4,000 | $0 |
| Google BigQuery | 1,000 | $0-5 |
| 商业API | 不限 | $200-2000 |

---

## 7. 参考资源

- [Google Patents Bulk Downloads](https://www.google.com/googlebooks/uspto-patents-grants-text.html)
- [USPTO Bulk Data](https://bulkdata.uspto.gov/)
- [EPO Bulk Data](https://www.epo.org/searching-for-patents/data/bulk-data-sets.html)
- [WIPO PATENTSCOPE](https://www.wipo.int/patentscope/en/)
- [The Lens API Docs](https://support.lens.org/support/solutions/articles/31000152654-the-lens-api)

---

*最后更新: 2026-04-24*
