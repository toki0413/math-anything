"""arXiv Client for Math Anything.

Provides access to mathematical and numerical methods literature.
No API key required. Searches abstracts and metadata only.

Privacy: Only mathematical fingerprints are used for queries,
never raw simulation data or proprietary parameters.
"""

import re
import urllib.request
import urllib.parse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET


@dataclass
class ArxivPaper:
    """Represent an arXiv paper."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    updated: str
    pdf_url: str
    relevance_score: float = 0.0
    matched_concepts: List[str] = None
    
    def __post_init__(self):
        if self.matched_concepts is None:
            self.matched_concepts = []


class ArxivClient:
    """Client for arXiv API queries.
    
    Optimized for mathematical and computational science literature.
    
    Example:
        ```python
        client = ArxivClient()
        
        # Query using mathematical fingerprint
        fingerprint = {
            "equation_patterns": ["[COEFF]*∇²u = [COEFF]*∂u/∂t"],
            "numerical_methods": ["Finite Element Method"],
        }
        
        papers = client.query_by_fingerprint(fingerprint, max_results=5)
        ```
    """
    
    API_BASE = "http://export.arxiv.org/api/query"
    
    # Math and CS categories relevant to computational science
    RELEVANT_CATEGORIES = [
        'math.NA',      # Numerical Analysis
        'math.AP',      # Analysis of PDEs
        'math-ph',      # Mathematical Physics
        'cs.NA',        # Numerical Analysis
        'cs.MS',        # Mathematical Software
        'cs.SC',        # Symbolic Computation
        'cond-mat.mtrl-sci',  # Materials Science
        'physics.chem-ph',    # Chemical Physics
        'physics.comp-ph',    # Computational Physics
        'stat.CO',      # Computation
    ]
    
    def __init__(self, cache=None):
        self.cache = cache
        self.last_query_time = None
        self.min_query_interval = 3.0  # arXiv rate limit
        
    def query_by_fingerprint(
        self,
        fingerprint: Dict[str, Any],
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> List[ArxivPaper]:
        """Query arXiv using mathematical fingerprint.
        
        Args:
            fingerprint: Mathematical fingerprint from schema
            max_results: Maximum number of results
            sort_by: Sorting criteria
            
        Returns:
            List of relevant arXiv papers
        """
        # Build query from fingerprint
        query = self._build_query_from_fingerprint(fingerprint)
        
        return self.search(query, max_results, sort_by)
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> List[ArxivPaper]:
        """Search arXiv with query string.
        
        Args:
            query: Search query
            max_results: Maximum results
            sort_by: Sorting method
            
        Returns:
            List of ArxivPaper objects
        """
        # Check cache first
        if self.cache:
            cache_key = f"arxiv:{query}:{max_results}:{sort_by}"
            cached = self.cache.get(cache_key)
            if cached:
                return [ArxivPaper(**p) for p in cached]
        
        # Build URL
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': 'descending',
        }
        
        url = f"{self.API_BASE}?{urllib.parse.urlencode(params)}"
        
        try:
            # Make request
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            
            # Parse response
            papers = self._parse_response(data)
            
            # Cache results
            if self.cache:
                cache_data = [self._paper_to_dict(p) for p in papers]
                self.cache.set(cache_key, cache_data, ttl=86400)  # 24 hours
            
            return papers
            
        except Exception as e:
            print(f"arXiv query error: {e}")
            return []
    
    def _build_query_from_fingerprint(self, fingerprint: Dict[str, Any]) -> str:
        """Build arXiv query from mathematical fingerprint."""
        query_parts = []
        
        # Add equation patterns
        for pattern in fingerprint.get("equation_patterns", [])[:3]:
            # Extract key terms from pattern
            terms = self._extract_math_terms(pattern)
            if terms:
                query_parts.append(f"({ ' OR '.join(terms) })")
        
        # Add numerical methods
        methods = fingerprint.get("numerical_methods", [])
        for method in methods[:2]:
            simplified = method.lower().replace('method', '').replace('finite', 'FEM')
            query_parts.append(f'"{simplified}"')
        
        # Add category filter
        cat_filter = " OR ".join([f"cat:{cat}" for cat in self.RELEVANT_CATEGORIES[:5]])
        
        query = " AND ".join(query_parts) if query_parts else "numerical simulation"
        query = f"({query}) AND ({cat_filter})"
        
        return query
    
    def _extract_math_terms(self, pattern: str) -> List[str]:
        """Extract searchable math terms from pattern."""
        terms = []
        
        # PDE types
        if any(x in pattern for x in ['∇²', 'laplacian', '∂²']):
            terms.extend(['Laplacian', 'elliptic', 'Poisson'])
        if '∂u/∂t' in pattern or '∂²u/∂t²' in pattern:
            terms.extend(['time-dependent', 'transient', 'parabolic'])
        
        # Operators
        if '∇·' in pattern or 'div' in pattern:
            terms.append('divergence')
        if '∇×' in pattern or 'curl' in pattern:
            terms.append('curl')
        
        # Remove duplicates
        return list(set(terms))
    
    def _parse_response(self, data: bytes) -> List[ArxivPaper]:
        """Parse arXiv API XML response."""
        papers = []
        
        try:
            root = ET.fromstring(data)
            
            # Define namespace
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            for entry in root.findall('atom:entry', ns):
                paper = self._parse_entry(entry, ns)
                if paper:
                    papers.append(paper)
                    
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            
        return papers
    
    def _parse_entry(self, entry: ET.Element, ns: Dict[str, str]) -> Optional[ArxivPaper]:
        """Parse single arXiv entry."""
        try:
            id_elem = entry.find('atom:id', ns)
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)
            updated_elem = entry.find('atom:updated', ns)
            
            # Get PDF link
            pdf_url = ""
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href', '')
                    break
            
            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Get categories
            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term', '')
                if term:
                    categories.append(term)
            
            return ArxivPaper(
                id=id_elem.text.split('/')[-1] if id_elem is not None else '',
                title=title_elem.text.strip() if title_elem is not None else '',
                authors=authors,
                abstract=summary_elem.text.strip() if summary_elem is not None else '',
                categories=categories,
                published=published_elem.text if published_elem is not None else '',
                updated=updated_elem.text if updated_elem is not None else '',
                pdf_url=pdf_url,
            )
            
        except Exception as e:
            print(f"Entry parse error: {e}")
            return None
    
    def _paper_to_dict(self, paper: ArxivPaper) -> Dict[str, Any]:
        """Convert paper to dictionary for caching."""
        return {
            'id': paper.id,
            'title': paper.title,
            'authors': paper.authors,
            'abstract': paper.abstract,
            'categories': paper.categories,
            'published': paper.published,
            'updated': paper.updated,
            'pdf_url': paper.pdf_url,
            'relevance_score': paper.relevance_score,
            'matched_concepts': paper.matched_concepts,
        }
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Get paper by arXiv ID."""
        papers = self.search(f"id:{arxiv_id}", max_results=1)
        return papers[0] if papers else None
