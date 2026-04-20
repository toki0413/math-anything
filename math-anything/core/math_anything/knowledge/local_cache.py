"""Local Knowledge Cache for Offline Availability.

Provides persistent local storage for knowledge base queries.
Ensures Math Anything remains functional offline.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pickle


class LocalKnowledgeCache:
    """Local cache for knowledge base responses.
    
    Features:
    - Persistent disk storage
    - TTL (time-to-live) support
    - Size limits with LRU eviction
    - Offline mode support
    
    Example:
        ```python
        cache = LocalKnowledgeCache()
        
        # Store with TTL
        cache.set("arxiv:query:123", data, ttl=86400)
        
        # Retrieve
        data = cache.get("arxiv:query:123")
        
        # Check offline mode
        if cache.is_offline_mode:
            print("Using cached data only")
        ```
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size_mb: int = 100,
        default_ttl: int = 86400,  # 24 hours
    ):
        """Initialize local cache.
        
        Args:
            cache_dir: Directory for cache files (default: ~/.math_anything/cache)
            max_size_mb: Maximum cache size in MB
            default_ttl: Default TTL in seconds
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.math_anything' / 'cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Offline mode flag
        self._offline_mode = False
        
    @property
    def is_offline_mode(self) -> bool:
        """Check if operating in offline mode."""
        return self._offline_mode
    
    def set_offline_mode(self, offline: bool = True):
        """Set offline mode.
        
        In offline mode, only cached data is returned.
        New queries will not attempt network requests.
        """
        self._offline_mode = offline
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return None
        
        # Check if expired
        if self._is_expired(key):
            self.delete(key)
            return None
        
        # Load data
        try:
            with open(cache_file, 'rb') as f:
                entry = pickle.load(f)
            
            # Update access time
            self.metadata[key]['last_accessed'] = datetime.now().isoformat()
            self._save_metadata()
            
            return entry['data']
            
        except Exception as e:
            print(f"Cache read error: {e}")
            return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None):
        """Store value in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Check cache size and evict if needed
        self._enforce_size_limit()
        
        cache_file = self._get_cache_file(key)
        
        entry = {
            'data': data,
            'created': datetime.now().isoformat(),
            'ttl': ttl,
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            
            # Update metadata
            self.metadata[key] = {
                'file': str(cache_file),
                'created': entry['created'],
                'ttl': ttl,
                'last_accessed': entry['created'],
                'size': cache_file.stat().st_size,
            }
            self._save_metadata()
            
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def delete(self, key: str):
        """Delete cached entry."""
        cache_file = self._get_cache_file(key)
        
        if cache_file.exists():
            cache_file.unlink()
        
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
    
    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob('*.cache'):
            cache_file.unlink()
        
        self.metadata = {}
        self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            Path(m['file']).stat().st_size
            for m in self.metadata.values()
            if Path(m['file']).exists()
        )
        
        # Count expired entries
        expired_count = sum(1 for key in self.metadata if self._is_expired(key))
        
        return {
            'num_entries': len(self.metadata),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'expired_entries': expired_count,
            'offline_mode': self._offline_mode,
            'cache_dir': str(self.cache_dir),
        }
    
    def cleanup_expired(self):
        """Remove all expired entries."""
        expired_keys = [key for key in self.metadata if self._is_expired(key)]
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash the key for filesystem safety
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, key: str) -> bool:
        """Check if cached entry is expired."""
        if key not in self.metadata:
            return True
        
        meta = self.metadata[key]
        created = datetime.fromisoformat(meta['created'])
        ttl = meta['ttl']
        
        return datetime.now() > created + timedelta(seconds=ttl)
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU eviction."""
        # Calculate current size
        current_size = sum(
            Path(m['file']).stat().st_size
            for m in self.metadata.values()
            if Path(m['file']).exists()
        )
        
        if current_size < self.max_size_bytes:
            return
        
        # Sort by last access time
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('last_accessed', x[1]['created'])
        )
        
        # Evict oldest entries until under limit
        for key, meta in sorted_entries:
            if current_size < self.max_size_bytes * 0.8:  # Target 80%
                break
            
            file_size = Path(meta['file']).stat().st_size
            self.delete(key)
            current_size -= file_size
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Metadata save error: {e}")
    
    def get_cache_keys(self, prefix: str = "") -> List[str]:
        """Get all cache keys matching prefix."""
        if prefix:
            return [k for k in self.metadata.keys() if k.startswith(prefix)]
        return list(self.metadata.keys())
