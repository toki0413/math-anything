"""Unit tests for Knowledge module: LocalKnowledgeCache, WikidataClient, ArxivClient."""

import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from math_anything.knowledge.arxiv_client import ArxivClient, ArxivPaper
from math_anything.knowledge.local_cache import LocalKnowledgeCache
from math_anything.knowledge.wikidata_client import WikidataClient, WikidataEntity

# ── LocalKnowledgeCache fixtures ──


@pytest.fixture
def cache(tmp_path):
    """Create a LocalKnowledgeCache in a temp directory."""
    return LocalKnowledgeCache(cache_dir=str(tmp_path), max_size_mb=10, default_ttl=3600)


@pytest.fixture
def cache_no_ttl(tmp_path):
    """Cache with very short TTL for expiration tests."""
    return LocalKnowledgeCache(cache_dir=str(tmp_path), default_ttl=1)


# ── LocalKnowledgeCache: creation ──


class TestLocalCacheCreation:
    def test_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "new_cache"
        LocalKnowledgeCache(cache_dir=str(cache_dir))
        assert cache_dir.exists()

    def test_default_cache_dir(self):
        c = LocalKnowledgeCache()
        assert c.cache_dir.exists()
        expected = Path.home() / ".math_anything" / "cache"
        assert c.cache_dir == expected

    def test_initial_metadata_empty(self, cache):
        assert cache.metadata == {}

    def test_offline_mode_default_false(self, cache):
        assert not cache.is_offline_mode


# ── LocalKnowledgeCache: set/get ──


class TestLocalCacheSetGet:
    def test_set_and_get(self, cache):
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        assert result == {"data": "value1"}

    def test_set_string_value(self, cache):
        cache.set("key2", "hello")
        assert cache.get("key2") == "hello"

    def test_set_numeric_value(self, cache):
        cache.set("key3", 42)
        assert cache.get("key3") == 42

    def test_set_list_value(self, cache):
        cache.set("key4", [1, 2, 3])
        assert cache.get("key4") == [1, 2, 3]

    def test_get_nonexistent_key(self, cache):
        assert cache.get("nonexistent") is None

    def test_overwrite_key(self, cache):
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_set_with_custom_ttl(self, cache):
        cache.set("short", "data", ttl=1)
        assert cache.get("short") == "data"


# ── LocalKnowledgeCache: expiration ──


class TestLocalCacheExpiration:
    def test_expired_entry_returns_none(self, cache_no_ttl):
        cache_no_ttl.set("expire_me", "data", ttl=0)
        # TTL=0 means it's already expired
        result = cache_no_ttl.get("expire_me")
        assert result is None

    def test_expired_entry_deleted(self, cache_no_ttl):
        cache_no_ttl.set("expire_me", "data", ttl=0)
        cache_no_ttl.get("expire_me")
        # The expired entry should be deleted from metadata
        assert "expire_me" not in cache_no_ttl.metadata

    def test_not_expired_entry_available(self, cache):
        cache.set("valid", "data", ttl=3600)
        assert cache.get("valid") == "data"


# ── LocalKnowledgeCache: delete/clear ──


class TestLocalCacheDeleteClear:
    def test_delete_key(self, cache):
        cache.set("key1", "data1")
        cache.delete("key1")
        assert cache.get("key1") is None
        assert "key1" not in cache.metadata

    def test_delete_nonexistent_key(self, cache):
        # Should not raise
        cache.delete("nonexistent")

    def test_clear_all(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.metadata == {}


# ── LocalKnowledgeCache: offline mode ──


class TestLocalCacheOfflineMode:
    def test_set_offline_mode(self, cache):
        cache.set_offline_mode(True)
        assert cache.is_offline_mode

    def test_unset_offline_mode(self, cache):
        cache.set_offline_mode(True)
        cache.set_offline_mode(False)
        assert not cache.is_offline_mode


# ── LocalKnowledgeCache: stats ──


class TestLocalCacheStats:
    def test_stats_empty(self, cache):
        stats = cache.get_stats()
        assert stats["num_entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["offline_mode"] is False

    def test_stats_with_entries(self, cache):
        cache.set("key1", "data1")
        stats = cache.get_stats()
        assert stats["num_entries"] == 1
        assert stats["total_size_bytes"] > 0

    def test_stats_max_size(self, cache):
        stats = cache.get_stats()
        assert stats["max_size_mb"] == 10


# ── LocalKnowledgeCache: cleanup ──


class TestLocalCacheCleanup:
    def test_cleanup_expired(self, cache):
        cache.set("valid", "data", ttl=3600)
        cache.set("expired", "data", ttl=0)
        count = cache.cleanup_expired()
        assert count == 1
        assert cache.get("valid") == "data"
        assert cache.get("expired") is None

    def test_cleanup_no_expired(self, cache):
        cache.set("valid", "data", ttl=3600)
        count = cache.cleanup_expired()
        assert count == 0


# ── LocalKnowledgeCache: get_cache_keys ──


class TestLocalCacheKeys:
    def test_get_all_keys(self, cache):
        cache.set("arxiv:1", "a")
        cache.set("wikidata:2", "b")
        cache.set("other:3", "c")
        keys = cache.get_cache_keys()
        assert len(keys) == 3

    def test_get_keys_with_prefix(self, cache):
        cache.set("arxiv:1", "a")
        cache.set("arxiv:2", "b")
        cache.set("wikidata:3", "c")
        keys = cache.get_cache_keys(prefix="arxiv")
        assert len(keys) == 2
        assert all(k.startswith("arxiv") for k in keys)

    def test_get_keys_empty(self, cache):
        keys = cache.get_cache_keys()
        assert keys == []


# ── LocalKnowledgeCache: file corruption ──


class TestLocalCacheCorruption:
    def test_corrupted_cache_file_returns_none(self, cache):
        cache.set("key1", "good_data")
        # Corrupt the cache file
        cache_file = cache._get_cache_file("key1")
        with open(cache_file, "wb") as f:
            f.write(b"corrupted data that is not pickle")
        result = cache.get("key1")
        assert result is None

    def test_corrupted_metadata_starts_fresh(self, tmp_path):
        # Write corrupted metadata
        meta_file = tmp_path / "cache_metadata.json"
        meta_file.write_text("not valid json{{{", encoding="utf-8")
        c = LocalKnowledgeCache(cache_dir=str(tmp_path))
        assert c.metadata == {}


# ── LocalKnowledgeCache: internal methods ──


class TestLocalCacheInternals:
    def test_get_cache_file_deterministic(self, cache):
        f1 = cache._get_cache_file("test_key")
        f2 = cache._get_cache_file("test_key")
        assert f1 == f2

    def test_get_cache_file_different_keys(self, cache):
        f1 = cache._get_cache_file("key1")
        f2 = cache._get_cache_file("key2")
        assert f1 != f2

    def test_is_expired_missing_key(self, cache):
        assert cache._is_expired("nonexistent") is True


# ── WikidataClient: creation ──


class TestWikidataClientCreation:
    def test_creates_without_cache(self):
        client = WikidataClient()
        assert client.cache is None

    def test_creates_with_cache(self, cache):
        client = WikidataClient(cache=cache)
        assert client.cache is not None

    def test_has_math_concepts(self):
        client = WikidataClient()
        assert "laplacian" in client.MATH_CONCEPTS
        assert "navier_stokes_equations" in client.MATH_CONCEPTS

    def test_has_units(self):
        client = WikidataClient()
        assert "pascal" in client.UNITS
        assert "meter" in client.UNITS

    def test_has_constants(self):
        client = WikidataClient()
        assert "speed_of_light" in client.CONSTANTS


# ── WikidataClient: get_concept (mocked network) ──


class TestWikidataClientConcept:
    def test_get_concept_from_cache(self, cache):
        # Pre-populate cache
        entity_data = {
            "id": "Q173165",
            "label": "Laplacian",
            "description": "differential operator",
            "properties": {},
            "uri": "http://www.wikidata.org/entity/Q173165",
        }
        cache.set("wikidata:concept:laplacian", entity_data)
        client = WikidataClient(cache=cache)
        result = client.get_concept("laplacian")
        assert result is not None
        assert result.label == "Laplacian"

    def test_get_concept_unknown_returns_none(self):
        client = WikidataClient()
        with patch.object(client, "_search_entity", return_value=None):
            result = client.get_concept("xyz_nonexistent_concept_12345")
            assert result is None

    def test_get_concept_normalizes_name(self):
        client = WikidataClient()
        # "Laplacian operator" -> "laplacian_operator" -> not in MATH_CONCEPTS
        # Should fall through to _search_entity
        with patch.object(client, "_search_entity", return_value=None) as mock:
            client.get_concept("Laplacian Operator")
            mock.assert_called_once()


# ── WikidataClient: get_unit ──


class TestWikidataClientUnit:
    def test_get_unit_from_cache(self, cache):
        entity_data = {
            "id": "Q44395",
            "label": "pascal",
            "description": "SI unit of pressure",
            "properties": {},
            "uri": "http://www.wikidata.org/entity/Q44395",
        }
        cache.set("wikidata:unit:pascal", entity_data)
        client = WikidataClient(cache=cache)
        result = client.get_unit("pascal")
        assert result is not None
        assert result.id == "Q44395"


# ── WikidataClient: get_constant ──


class TestWikidataClientConstant:
    def test_get_constant_known(self):
        client = WikidataClient()
        with patch.object(client, "_get_entity_by_id") as mock:
            mock.return_value = WikidataEntity(
                id="Q2111",
                label="speed of light",
                description="physical constant",
                properties={},
                uri="http://www.wikidata.org/entity/Q2111",
            )
            result = client.get_constant("speed_of_light")
            assert result is not None
            assert result.id == "Q2111"


# ── WikidataClient: get_equation_info ──


class TestWikidataClientEquation:
    def test_get_equation_info_known(self):
        client = WikidataClient()
        with patch.object(client, "_get_entity_by_id") as mock:
            mock.return_value = WikidataEntity(
                id="Q2000011",
                label="Navier-Stokes",
                description="equations",
                properties={},
                uri="http://www.wikidata.org/entity/Q2000011",
            )
            result = client.get_equation_info("navier-stokes")
            assert "name" in result
            assert result["wikidata_id"] == "Q2000011"

    def test_get_equation_info_unknown(self):
        client = WikidataClient()
        with patch.object(client, "_get_entity_by_id", return_value=None):
            result = client.get_equation_info("unknown_equation_xyz")
            assert result == {}


# ── WikidataClient: get_numerical_method_info ──


class TestWikidataClientMethod:
    def test_get_numerical_method_fem(self):
        client = WikidataClient()
        with patch.object(client, "_get_entity_by_id") as mock:
            mock.return_value = WikidataEntity(
                id="Q236425",
                label="FEM",
                description="numerical method",
                properties={},
                uri="http://www.wikidata.org/entity/Q236425",
            )
            result = client.get_numerical_method_info("finite element")
            assert result["wikidata_id"] == "Q236425"

    def test_get_numerical_method_unknown(self):
        client = WikidataClient()
        with patch.object(client, "_get_entity_by_id", return_value=None):
            result = client.get_numerical_method_info("unknown_method_xyz")
            assert result == {}


# ── WikidataClient: query_by_fingerprint ──


class TestWikidataClientFingerprint:
    def test_query_by_fingerprint(self):
        client = WikidataClient()
        with (
            patch.object(client, "get_concept") as mock_concept,
            patch.object(client, "get_numerical_method_info") as mock_method,
        ):
            mock_concept.return_value = WikidataEntity(
                id="Q1", label="Test", description="test", properties={}, uri="http://www.wikidata.org/entity/Q1"
            )
            mock_method.return_value = {
                "name": "FEM",
                "description": "method",
                "wikidata_id": "Q236425",
                "uri": "http://test",
            }
            result = client.query_by_fingerprint(
                {"mathematical_concepts": ["laplacian"], "tensor_operations": ["finite element"]}
            )
            assert len(result) == 2

    def test_query_by_fingerprint_empty(self):
        client = WikidataClient()
        result = client.query_by_fingerprint({})
        assert result == []


# ── WikidataClient: network error handling ──


class TestWikidataClientNetworkErrors:
    def test_get_entity_by_id_network_error(self):
        client = WikidataClient()
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            result = client._get_entity_by_id("Q12345")
            assert result is None

    def test_search_entity_network_error(self):
        client = WikidataClient()
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            result = client._search_entity("test")
            assert result is None

    def test_execute_sparql_network_error(self):
        client = WikidataClient()
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            result = client._execute_sparql("SELECT ?x WHERE { ?x ?y ?z }")
            assert result == []


# ── WikidataEntity ──


class TestWikidataEntity:
    def test_entity_creation(self):
        entity = WikidataEntity(
            id="Q123",
            label="Test",
            description="A test entity",
            properties={"p1": "v1"},
            uri="http://www.wikidata.org/entity/Q123",
        )
        assert entity.id == "Q123"
        assert entity.label == "Test"


# ── ArxivClient: creation ──


class TestArxivClientCreation:
    def test_creates_without_cache(self):
        client = ArxivClient()
        assert client.cache is None
        assert client.min_query_interval == 3.0

    def test_creates_with_cache(self, cache):
        client = ArxivClient(cache=cache)
        assert client.cache is not None

    def test_has_relevant_categories(self):
        client = ArxivClient()
        assert "math.NA" in client.RELEVANT_CATEGORIES
        assert "physics.comp-ph" in client.RELEVANT_CATEGORIES


# ── ArxivClient: search (mocked network) ──


class TestArxivClientSearch:
    def test_search_with_mock_response(self):
        client = ArxivClient()
        # Build a minimal arXiv XML response
        xml_response = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/2301.00001</id>
            <title>Test Paper Title</title>
            <summary>A test abstract about numerical methods.</summary>
            <published>2023-01-01T00:00:00Z</published>
            <updated>2023-01-02T00:00:00Z</updated>
            <author><name>Author One</name></author>
            <category term="math.NA"/>
            <link title="pdf" href="http://arxiv.org/pdf/2301.00001"/>
          </entry>
        </feed>"""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = xml_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            papers = client.search("numerical methods", max_results=1)
            assert len(papers) == 1
            assert papers[0].title == "Test Paper Title"
            assert papers[0].id == "2301.00001"
            assert "Author One" in papers[0].authors
            assert "math.NA" in papers[0].categories

    def test_search_network_error(self):
        client = ArxivClient()
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            papers = client.search("test query")
            assert papers == []

    def test_search_with_cache_hit(self, cache):
        # Pre-populate cache
        cached_data = [
            {
                "id": "2301.00001",
                "title": "Cached Paper",
                "authors": ["Author"],
                "abstract": "Abstract",
                "categories": ["math.NA"],
                "published": "2023-01-01",
                "updated": "2023-01-01",
                "pdf_url": "http://arxiv.org/pdf/2301.00001",
                "relevance_score": 0.0,
                "matched_concepts": [],
            }
        ]
        cache.set("arxiv:test query:1:relevance", cached_data)
        client = ArxivClient(cache=cache)
        papers = client.search("test query", max_results=1)
        assert len(papers) == 1
        assert papers[0].title == "Cached Paper"


# ── ArxivClient: query_by_fingerprint ──


class TestArxivClientFingerprint:
    def test_query_by_fingerprint(self):
        client = ArxivClient()
        with patch.object(client, "search") as mock_search:
            mock_search.return_value = []
            fingerprint = {
                "equation_patterns": ["∇²u = f"],
                "numerical_methods": ["Finite Element Method"],
            }
            client.query_by_fingerprint(fingerprint, max_results=5)
            mock_search.assert_called_once()
            # Check the query contains FEM
            call_args = mock_search.call_args
            assert "FEM" in call_args[0][0] or "FEM" in call_args[1].get("query", "")


# ── ArxivClient: _build_query_from_fingerprint ──


class TestArxivClientBuildQuery:
    def test_build_query_with_methods(self):
        client = ArxivClient()
        query = client._build_query_from_fingerprint({"numerical_methods": ["Finite Element Method"]})
        assert "FEM" in query

    def test_build_query_empty_fingerprint(self):
        client = ArxivClient()
        query = client._build_query_from_fingerprint({})
        assert "numerical simulation" in query


# ── ArxivClient: _extract_math_terms ──


class TestArxivClientMathTerms:
    def test_extract_laplacian_terms(self):
        client = ArxivClient()
        terms = client._extract_math_terms("∇²u = f(x)")
        assert "Laplacian" in terms

    def test_extract_time_dependent_terms(self):
        client = ArxivClient()
        terms = client._extract_math_terms("∂u/∂t = α∇²u")
        assert "time-dependent" in terms

    def test_extract_divergence_terms(self):
        client = ArxivClient()
        terms = client._extract_math_terms("∇·F = 0")
        assert "divergence" in terms

    def test_extract_no_terms(self):
        client = ArxivClient()
        terms = client._extract_math_terms("simple text without math")
        assert terms == []


# ── ArxivClient: get_paper_by_id ──


class TestArxivClientGetById:
    def test_get_paper_by_id(self):
        client = ArxivClient()
        with patch.object(client, "search") as mock_search:
            mock_search.return_value = [
                ArxivPaper(
                    id="2301.00001",
                    title="Test",
                    authors=["A"],
                    abstract="abs",
                    categories=["math.NA"],
                    published="2023-01-01",
                    updated="2023-01-01",
                    pdf_url="http://test",
                )
            ]
            paper = client.get_paper_by_id("2301.00001")
            assert paper is not None
            assert paper.id == "2301.00001"

    def test_get_paper_by_id_not_found(self):
        client = ArxivClient()
        with patch.object(client, "search", return_value=[]):
            paper = client.get_paper_by_id("nonexistent")
            assert paper is None


# ── ArxivPaper ──


class TestArxivPaper:
    def test_paper_creation(self):
        paper = ArxivPaper(
            id="2301.00001",
            title="Test",
            authors=["A"],
            abstract="abs",
            categories=["math.NA"],
            published="2023-01-01",
            updated="2023-01-01",
            pdf_url="http://test",
        )
        assert paper.id == "2301.00001"
        assert paper.matched_concepts == []  # default from __post_init__

    def test_paper_to_dict(self):
        client = ArxivClient()
        paper = ArxivPaper(
            id="2301.00001",
            title="Test",
            authors=["A"],
            abstract="abs",
            categories=["math.NA"],
            published="2023-01-01",
            updated="2023-01-01",
            pdf_url="http://test",
            relevance_score=0.8,
            matched_concepts=["FEM"],
        )
        d = client._paper_to_dict(paper)
        assert d["id"] == "2301.00001"
        assert d["relevance_score"] == 0.8
        assert d["matched_concepts"] == ["FEM"]


# ── ArxivClient: parse_response ──


class TestArxivClientParse:
    def test_parse_invalid_xml(self):
        client = ArxivClient()
        result = client._parse_response(b"not xml at all")
        assert result == []

    def test_parse_empty_feed(self):
        client = ArxivClient()
        xml = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        result = client._parse_response(xml)
        assert result == []
