"""
HNQT v7: Comprehensive Evaluation with Aggressive Search Configs
=================================================================
Key improvements over v5/v6:
1. Expanded search configurations from minimal to exhaustive
2. Cleaned up dataset support (BioASQ only for rigor)
3. Better metrics: shows % of index searched
4. Sorted results from least to most aggressive
5. Summary statistics for recall/efficiency tradeoffs
"""

import json
import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import pickle
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import time
import shutil


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CrossLink:
    """Represents a teleportation link between L2 clusters"""
    target_l1: int
    target_l2: int
    similarity: float

@dataclass 
class RouterInfo:
    """Information stored in each directory's router"""
    centroid: np.ndarray
    child_centroids: Dict[int, np.ndarray] = field(default_factory=dict)
    cross_links: List[CrossLink] = field(default_factory=list)
    vector_count: int = 0


# ============================================================================
# VECTOR QUANTIZER WITH TELEPORTATION
# ============================================================================

class HNQTQuantizer:
    """
    Hierarchical Navigable Quantization Tree
    
    Two-level hierarchy with cross-links for teleportation:
    - L1: Coarse partitioning (4-8 clusters)
    - L2: Fine partitioning within each L1 (4-8 clusters per L1)
    - Cross-links: Connections between semantically similar L2 clusters
    """
    
    def __init__(self, n_clusters_l1: int = 4, n_clusters_l2: int = 4):
        self.n_clusters_l1 = n_clusters_l1
        self.n_clusters_l2 = n_clusters_l2
        
        # Trained models
        self.l1_kmeans: Optional[KMeans] = None
        self.l2_kmeans: Dict[int, Optional[KMeans]] = {}
        
        # Hex code mappings
        self.l1_codes: List[str] = []
        self.l2_codes: Dict[int, List[str]] = {}
        
        # Cross-links for teleportation: (l1, l2) -> [CrossLink, ...]
        self.cross_links: Dict[Tuple[int, int], List[CrossLink]] = {}
        
        # Router info cache
        self.routers: Dict[str, RouterInfo] = {}
        
    def train(self, embeddings: np.ndarray, build_cross_links: bool = True):
        """Train the hierarchical quantizer"""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        n_samples = len(embeddings)
        
        print(f"Training HNQT quantizer on {n_samples} vectors...")
        print(f"  L1 clusters: {self.n_clusters_l1}")
        print(f"  L2 clusters per L1: {self.n_clusters_l2}")
        print(f"  Total buckets: {self.n_clusters_l1 * self.n_clusters_l2}")
        
        # ===== Level 1: Coarse Clustering =====
        print("\nTraining Level 1 (coarse)...")
        self.l1_kmeans = KMeans(
            n_clusters=self.n_clusters_l1, 
            random_state=42, 
            n_init=10
        )
        self.l1_kmeans.fit(embeddings)
        self.l1_codes = [format(i, '02x') for i in range(self.n_clusters_l1)]
        
        # Store L1 router info
        root_router = RouterInfo(
            centroid=np.mean(embeddings, axis=0),
            child_centroids={i: self.l1_kmeans.cluster_centers_[i] 
                           for i in range(self.n_clusters_l1)},
            vector_count=n_samples
        )
        self.routers["root"] = root_router
        
        # ===== Level 2: Fine Clustering within each L1 =====
        print("Training Level 2 (fine)...")
        
        for l1_idx in tqdm(range(self.n_clusters_l1), desc="L2 clusters"):
            mask = self.l1_kmeans.labels_ == l1_idx
            cluster_embeddings = embeddings[mask]
            n_cluster = len(cluster_embeddings)
            
            # Determine appropriate number of L2 clusters
            actual_l2_clusters = min(self.n_clusters_l2, max(1, n_cluster // 2))
            
            if n_cluster < 2:
                # Degenerate case: single vector
                self.l2_kmeans[l1_idx] = None
                self.l2_codes[l1_idx] = ["00"]
                continue
            
            try:
                l2_model = KMeans(
                    n_clusters=actual_l2_clusters,
                    random_state=42 + l1_idx,
                    n_init=5
                )
                l2_model.fit(cluster_embeddings)
                self.l2_kmeans[l1_idx] = l2_model
                self.l2_codes[l1_idx] = [format(i, '02x') for i in range(actual_l2_clusters)]
                
                # Store L1 router info
                l1_router = RouterInfo(
                    centroid=self.l1_kmeans.cluster_centers_[l1_idx],
                    child_centroids={i: l2_model.cluster_centers_[i] 
                                   for i in range(actual_l2_clusters)},
                    vector_count=n_cluster
                )
                self.routers[self.l1_codes[l1_idx]] = l1_router
                
            except Exception as e:
                print(f"  Warning: L2 clustering failed for L1={l1_idx}: {e}")
                self.l2_kmeans[l1_idx] = None
                self.l2_codes[l1_idx] = ["00"]
        
        # ===== Build Cross-Links for Teleportation =====
        if build_cross_links:
            self._build_cross_links()
        
        print(f"\nQuantizer trained successfully!")
    
    def _build_cross_links(self, top_k: int = 3, min_similarity: float = 0.3):
        """Build teleportation links between L2 clusters in different L1 branches"""
        print("\nBuilding cross-links for teleportation...")
        
        # Collect all L2 centroids
        l2_info = []  # [(l1_idx, l2_idx, centroid), ...]
        
        for l1_idx in range(self.n_clusters_l1):
            l2_model = self.l2_kmeans.get(l1_idx)
            if l2_model is None:
                # Single cluster case
                l2_info.append((l1_idx, 0, self.l1_kmeans.cluster_centers_[l1_idx]))
            else:
                for l2_idx in range(l2_model.n_clusters):
                    l2_info.append((l1_idx, l2_idx, l2_model.cluster_centers_[l2_idx]))
        
        # For each L2 cluster, find similar clusters in OTHER L1 branches
        total_links = 0
        for i, (l1_i, l2_i, centroid_i) in enumerate(l2_info):
            similarities = []
            
            for j, (l1_j, l2_j, centroid_j) in enumerate(l2_info):
                if l1_i == l1_j:  # Skip same L1 branch
                    continue
                
                # Cosine similarity
                norm_i = np.linalg.norm(centroid_i)
                norm_j = np.linalg.norm(centroid_j)
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(centroid_i, centroid_j) / (norm_i * norm_j)
                else:
                    sim = 0.0
                
                if sim >= min_similarity:
                    similarities.append(CrossLink(l1_j, l2_j, sim))
            
            # Keep top-k cross-links
            similarities.sort(key=lambda x: -x.similarity)
            self.cross_links[(l1_i, l2_i)] = similarities[:top_k]
            total_links += len(similarities[:top_k])
        
        print(f"  Built {total_links} cross-links across {len([k for k,v in self.cross_links.items() if v])} clusters")
    
    def quantize(self, embedding: np.ndarray) -> Tuple[str, str, int, int]:
        """
        Quantize a vector to (l1_code, l2_code, l1_idx, l2_idx)
        
        Returns both hex codes (for filesystem) and indices (for cross-links)
        """
        embedding = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        
        # L1 quantization
        l1_idx = self.l1_kmeans.predict(embedding)[0]
        l1_code = self.l1_codes[l1_idx]
        
        # L2 quantization
        l2_model = self.l2_kmeans.get(l1_idx)
        if l2_model is not None:
            l2_idx = l2_model.predict(embedding)[0]
            # Handle case where prediction exceeds available codes
            if l2_idx >= len(self.l2_codes[l1_idx]):
                l2_idx = 0
            l2_code = self.l2_codes[l1_idx][l2_idx]
        else:
            l2_idx = 0
            l2_code = "00"
        
        return l1_code, l2_code, l1_idx, l2_idx
    
    def get_l1_distances(self, embedding: np.ndarray) -> np.ndarray:
        """Get distances to all L1 centroids"""
        embedding = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        return self.l1_kmeans.transform(embedding)[0]
    
    def get_l2_distances(self, embedding: np.ndarray, l1_idx: int) -> np.ndarray:
        """Get distances to all L2 centroids within an L1 cluster"""
        embedding = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        l2_model = self.l2_kmeans.get(l1_idx)
        if l2_model is not None:
            return l2_model.transform(embedding)[0]
        else:
            return np.array([0.0])
    
    def get_cross_links(self, l1_idx: int, l2_idx: int) -> List[CrossLink]:
        """Get teleportation links for a given L2 cluster"""
        return self.cross_links.get((l1_idx, l2_idx), [])
    
    def get_total_buckets(self) -> int:
        """Get total number of L2 buckets"""
        return sum(len(codes) for codes in self.l2_codes.values())


# ============================================================================
# HNQT INDEX
# ============================================================================

class HNQTIndex:
    """
    Hierarchical Navigable Quantization Tree Index
    
    Features:
    - Two-level hierarchy with SQLite leaf databases
    - Multi-probe search with configurable beam width
    - Cross-link teleportation for improved recall
    - Comparison against FAISS baseline
    """
    
    def __init__(self, base_path: str = "vdb_v7", n_clusters_l1: int = 4, n_clusters_l2: int = 4):
        self.base_path = Path(base_path)
        
        # Clear existing index
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.quantizer = HNQTQuantizer(n_clusters_l1, n_clusters_l2)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # FAISS baseline for comparison
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.n_docs = 0
        
        # Metrics tracking
        self.metrics = {
            "insert_times": [],
            "search_times": [],
            "buckets_searched": [],
            "teleports_used": []
        }
    
    def build(self, texts: List[str], embeddings: Optional[np.ndarray] = None):
        """Build the HNQT index from texts"""
        
        # Generate embeddings if not provided
        if embeddings is None:
            print("Generating embeddings...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
        
        embeddings = np.asarray(embeddings, dtype=np.float64)
        self.n_docs = len(texts)
        
        # Train quantizer
        self.quantizer.train(embeddings)
        
        # Build FAISS baseline
        print("\nBuilding FAISS baseline...")
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # Insert into HNQT
        print("\nInserting into HNQT...")
        for i, (text, emb) in enumerate(tqdm(zip(texts, embeddings), 
                                             total=len(texts), desc="Inserting")):
            start = time.time()
            self._insert_single(i, text, emb)
            self.metrics["insert_times"].append(time.time() - start)
        
        print(f"\nIndex built successfully!")
        print(f"  Total documents: {self.n_docs}")
        print(f"  Total buckets: {self.quantizer.get_total_buckets()}")
        print(f"  Average insert time: {np.mean(self.metrics['insert_times']):.4f}s")
        self._print_structure()
    
    def _insert_single(self, idx: int, text: str, embedding: np.ndarray):
        """Insert a single document into the index"""
        l1_code, l2_code, _, _ = self.quantizer.quantize(embedding)
        
        # Create directory path
        leaf_path = self.base_path / l1_code / l2_code
        leaf_path.mkdir(parents=True, exist_ok=True)
        
        # Insert into SQLite
        db_path = leaf_path / "data.db"
        
        conn = sqlite3.connect(str(db_path))
        
        # Create table if needed
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY,
                orig_index INTEGER,
                text TEXT,
                embedding BLOB
            )
        """)
        
        # Insert
        conn.execute(
            "INSERT INTO vectors (orig_index, text, embedding) VALUES (?, ?, ?)",
            (idx, text, embedding.astype(np.float32).tobytes())
        )
        conn.commit()
        conn.close()
    
    def _print_structure(self):
        """Print the index structure"""
        print("\nIndex Structure:")
        total_vectors = 0
        
        for l1_dir in sorted(self.base_path.iterdir()):
            if not l1_dir.is_dir():
                continue
            
            l1_count = 0
            l2_info = []
            
            for l2_dir in sorted(l1_dir.iterdir()):
                if not l2_dir.is_dir():
                    continue
                
                db_path = l2_dir / "data.db"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
                    conn.close()
                    l2_info.append(f"{l2_dir.name}({count})")
                    l1_count += count
            
            total_vectors += l1_count
            print(f"  {l1_dir.name}/: {l1_count} vectors [{', '.join(l2_info)}]")
        
        print(f"  Total: {total_vectors} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 10,
               n_probe_l1: int = 2, n_probe_l2: int = 3,
               use_teleport: bool = True, teleport_threshold: float = 0.3) -> Tuple[List[int], Dict]:
        """
        Search the HNQT index with multi-probe and teleportation
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            n_probe_l1: Number of L1 clusters to probe
            n_probe_l2: Number of L2 clusters to probe per L1
            use_teleport: Whether to use cross-link teleportation
            teleport_threshold: Minimum similarity for teleportation
        
        Returns:
            (result_indices, search_stats)
        """
        start = time.time()
        
        query_embedding = np.asarray(query_embedding, dtype=np.float64).flatten()
        
        candidates = []  # [(distance, orig_index), ...]
        visited_buckets: Set[Tuple[int, int]] = set()
        buckets_to_search: List[Tuple[int, int, float]] = []  # (l1_idx, l2_idx, priority)
        teleports_used = 0
        
        # ===== Phase 1: Identify initial buckets to search =====
        
        # Get top L1 clusters
        l1_distances = self.quantizer.get_l1_distances(query_embedding)
        l1_top_indices = np.argsort(l1_distances)[:n_probe_l1]
        
        for l1_idx in l1_top_indices:
            # Get top L2 clusters within this L1
            l2_distances = self.quantizer.get_l2_distances(query_embedding, l1_idx)
            l2_top_indices = np.argsort(l2_distances)[:n_probe_l2]
            
            for l2_idx in l2_top_indices:
                if l2_idx < len(self.quantizer.l2_codes[l1_idx]):
                    priority = l1_distances[l1_idx] + l2_distances[l2_idx]
                    buckets_to_search.append((l1_idx, l2_idx, priority))
        
        # ===== Phase 2: Search buckets with teleportation =====
        
        # Sort by priority (lower distance = higher priority)
        buckets_to_search.sort(key=lambda x: x[2])
        
        for l1_idx, l2_idx, _ in buckets_to_search:
            if (l1_idx, l2_idx) in visited_buckets:
                continue
            
            visited_buckets.add((l1_idx, l2_idx))
            
            # Search this bucket
            bucket_results = self._search_bucket(l1_idx, l2_idx, query_embedding)
            candidates.extend(bucket_results)
            
            # Teleportation: follow cross-links to related buckets
            if use_teleport:
                cross_links = self.quantizer.get_cross_links(l1_idx, l2_idx)
                for link in cross_links:
                    if link.similarity >= teleport_threshold:
                        target = (link.target_l1, link.target_l2)
                        if target not in visited_buckets:
                            visited_buckets.add(target)
                            teleport_results = self._search_bucket(
                                link.target_l1, link.target_l2, query_embedding
                            )
                            candidates.extend(teleport_results)
                            teleports_used += 1
        
        # ===== Phase 3: Merge and deduplicate results =====
        
        candidates.sort(key=lambda x: x[0])
        
        seen = set()
        results = []
        for dist, idx in candidates:
            if idx not in seen:
                seen.add(idx)
                results.append(idx)
                if len(results) >= k:
                    break
        
        search_time = time.time() - start
        
        stats = {
            "search_time": search_time,
            "buckets_searched": len(visited_buckets),
            "teleports_used": teleports_used,
            "candidates_found": len(candidates)
        }
        
        return results, stats
    
    def _search_bucket(self, l1_idx: int, l2_idx: int, 
                       query_embedding: np.ndarray) -> List[Tuple[float, int]]:
        """Search a single L2 bucket, return [(distance, orig_index), ...]"""
        l1_code = self.quantizer.l1_codes[l1_idx]
        
        if l2_idx >= len(self.quantizer.l2_codes[l1_idx]):
            return []
        
        l2_code = self.quantizer.l2_codes[l1_idx][l2_idx]
        
        db_path = self.base_path / l1_code / l2_code / "data.db"
        
        if not db_path.exists():
            return []
        
        results = []
        conn = sqlite3.connect(str(db_path))
        
        cursor = conn.execute("SELECT orig_index, embedding FROM vectors")
        for orig_idx, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32).astype(np.float64)
            dist = np.linalg.norm(query_embedding - emb)
            results.append((dist, orig_idx))
        
        conn.close()
        return results
    
    def search_faiss(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[int], float]:
        """Search FAISS baseline"""
        start = time.time()
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.faiss_index.search(query, k)
        search_time = time.time() - start
        return indices[0].tolist(), search_time


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

class HNQTEvaluator:
    """Comprehensive evaluation of HNQT vs FAISS baseline"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_bioasq(self, max_docs: int = 1000, max_queries: int = 100) -> Dict[str, Any]:
        """
        Load BioASQ biomedical QA dataset.
        
        Uses embedding-based ground truth (FAISS brute-force) for fair comparison.
        """
        from datasets import load_dataset
        import ast
        
        print(f"Loading BioASQ dataset (max_docs={max_docs}, max_queries={max_queries})...")
        
        # Load corpus
        print("  Loading corpus...")
        corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
        
        # Load QA pairs
        print("  Loading questions...")
        qa = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test")
        
        # Process documents
        docs_data = []
        passage_id_map = {}
        
        for idx, p in enumerate(tqdm(corpus, desc="Processing corpus")):
            if len(docs_data) >= max_docs:
                break
            text = p.get("passage", "") or ""
            if not text or len(text) < 20:
                continue
            doc_id = f"bioasq_doc_{idx}"
            docs_data.append({"id": doc_id, "text": text})
            if p.get("id") is not None:
                passage_id_map[p["id"]] = doc_id
        
        # Process queries
        queries_data = []
        for idx, q in enumerate(tqdm(qa, desc="Processing queries")):
            if len(queries_data) >= max_queries:
                break
            text = q.get("question", "")
            if not text:
                continue
            
            # Map passage IDs
            rel_ids = q.get("relevant_passage_ids", [])
            if isinstance(rel_ids, str):
                try:
                    rel_ids = ast.literal_eval(rel_ids)
                except:
                    rel_ids = []
            
            expected = [passage_id_map[pid] for pid in rel_ids if pid in passage_id_map]
            
            queries_data.append({
                "id": f"bioasq_q_{idx}",
                "text": text,
                "expected_doc_ids": expected,
                "answer": q.get("answer", "")
            })
        
        print(f"Loaded {len(docs_data)} documents, {len(queries_data)} queries")
        
        # Generate embeddings
        print("Generating document embeddings...")
        doc_texts = [d["text"] for d in docs_data]
        doc_embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        
        print("Generating query embeddings...")
        query_texts = [q["text"] for q in queries_data]
        query_embeddings = self.model.encode(query_texts, show_progress_bar=True)
        
        # Compute embedding-based ground truth
        print("Computing embedding-based ground truth...")
        dim = doc_embeddings.shape[1]
        brute_force_index = faiss.IndexFlatL2(dim)
        brute_force_index.add(doc_embeddings.astype(np.float32))
        
        ground_truth_k = 10
        ground_truth = []
        
        for query_emb in query_embeddings:
            distances, indices = brute_force_index.search(
                query_emb.astype(np.float32).reshape(1, -1), 
                ground_truth_k
            )
            ground_truth.append(indices[0].tolist())
        
        print(f"Ground truth computed: {ground_truth_k} nearest neighbors per query")
        
        return {
            "name": "bioasq",
            "documents": doc_texts,
            "queries": query_texts,
            "doc_embeddings": doc_embeddings,
            "query_embeddings": query_embeddings,
            "ground_truth": ground_truth,
        }
    
    def create_synthetic_dataset(self, n_docs: int = 1000, n_queries: int = 50) -> Dict[str, Any]:
        """Create synthetic dataset for quick testing"""
        print(f"Creating synthetic dataset: {n_docs} docs, {n_queries} queries...")
        
        topics = [
            "machine learning algorithms and neural networks",
            "natural language processing and text analysis", 
            "computer vision and image recognition",
            "reinforcement learning and decision making",
            "database systems and query optimization",
            "operating systems and process management",
            "distributed systems and cloud computing",
            "compiler design and code optimization",
        ]
        
        documents = []
        templates = [
            "A comprehensive guide to {}. This document explores {} in depth.",
            "Research paper on {}. We present novel findings about {}.",
            "Tutorial: Getting started with {}. Learn the basics of {}.",
            "Case study: Applying {} in industry. Real-world examples of {}.",
        ]
        
        for i in range(n_docs):
            topic = topics[i % len(topics)]
            template = templates[i % len(templates)]
            documents.append(template.format(topic, topic))
        
        queries = []
        query_templates = ["Find information about {}", "What is {}?", "Explain {}"]
        for q in range(n_queries):
            topic = topics[q % len(topics)]
            template = query_templates[q % len(query_templates)]
            queries.append(template.format(topic))
        
        print("Generating embeddings...")
        doc_embeddings = self.model.encode(documents, show_progress_bar=True)
        query_embeddings = self.model.encode(queries, show_progress_bar=True)
        
        # Compute ground truth
        print("Computing ground truth...")
        dim = doc_embeddings.shape[1]
        brute_force_index = faiss.IndexFlatL2(dim)
        brute_force_index.add(doc_embeddings.astype(np.float32))
        
        ground_truth = []
        for query_emb in query_embeddings:
            distances, indices = brute_force_index.search(
                query_emb.astype(np.float32).reshape(1, -1), 10
            )
            ground_truth.append(indices[0].tolist())
        
        return {
            "name": "synthetic",
            "documents": documents,
            "queries": queries,
            "doc_embeddings": doc_embeddings,
            "query_embeddings": query_embeddings,
            "ground_truth": ground_truth,
        }
    
    def get_search_configs(self, n_l1: int, n_l2: int) -> List[Dict]:
        """
        Generate comprehensive search configurations from least to most aggressive.
        
        Configs are ordered by expected recall (ascending).
        """
        total_buckets = n_l1 * n_l2
        
        configs = [
            # === Minimal Search ===
            {"n_probe_l1": 1, "n_probe_l2": 1, "use_teleport": False, 
             "name": "HNQT (1,1) no teleport"},
            
            {"n_probe_l1": 1, "n_probe_l2": 1, "use_teleport": True, 
             "name": "HNQT (1,1) + teleport"},
            
            # === Light Search ===
            {"n_probe_l1": 1, "n_probe_l2": 2, "use_teleport": False, 
             "name": "HNQT (1,2) no teleport"},
            
            {"n_probe_l1": 1, "n_probe_l2": 2, "use_teleport": True, 
             "name": "HNQT (1,2) + teleport"},
            
            {"n_probe_l1": 2, "n_probe_l2": 2, "use_teleport": False, 
             "name": "HNQT (2,2) no teleport"},
            
            {"n_probe_l1": 2, "n_probe_l2": 2, "use_teleport": True, 
             "name": "HNQT (2,2) + teleport"},
            
            # === Medium Search ===
            {"n_probe_l1": 2, "n_probe_l2": 3, "use_teleport": False, 
             "name": "HNQT (2,3) no teleport"},
            
            {"n_probe_l1": 2, "n_probe_l2": 3, "use_teleport": True, 
             "name": "HNQT (2,3) + teleport"},
            
            {"n_probe_l1": 3, "n_probe_l2": 3, "use_teleport": False, 
             "name": "HNQT (3,3) no teleport"},
            
            {"n_probe_l1": 3, "n_probe_l2": 3, "use_teleport": True, 
             "name": "HNQT (3,3) + teleport"},
            
            # === Aggressive Search ===
            {"n_probe_l1": 3, "n_probe_l2": 4, "use_teleport": True, 
             "name": "HNQT (3,4) + teleport"},
            
            {"n_probe_l1": 4, "n_probe_l2": 4, "use_teleport": True, 
             "name": "HNQT (4,4) + teleport"},
            
            # === Very Aggressive Search ===
            {"n_probe_l1": 4, "n_probe_l2": 5, "use_teleport": True, 
             "name": "HNQT (4,5) + teleport"},
            
            {"n_probe_l1": 5, "n_probe_l2": 5, "use_teleport": True, 
             "name": "HNQT (5,5) + teleport"},
            
            # === Near-Exhaustive Search ===
            {"n_probe_l1": 6, "n_probe_l2": 6, "use_teleport": True, 
             "name": "HNQT (6,6) + teleport"},
            
            {"n_probe_l1": min(8, n_l1), "n_probe_l2": min(8, n_l2), "use_teleport": True, 
             "name": f"HNQT ({min(8, n_l1)},{min(8, n_l2)}) + teleport [max]"},
        ]
        
        # Filter configs that exceed cluster counts
        valid_configs = []
        for cfg in configs:
            if cfg["n_probe_l1"] <= n_l1 and cfg["n_probe_l2"] <= n_l2:
                valid_configs.append(cfg)
        
        return valid_configs
    
    def evaluate(self, index: HNQTIndex, dataset: Dict[str, Any], k: int = 10,
                 search_configs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        
        n_l1 = index.quantizer.n_clusters_l1
        n_l2 = index.quantizer.n_clusters_l2
        total_buckets = index.quantizer.get_total_buckets()
        
        if search_configs is None:
            search_configs = self.get_search_configs(n_l1, n_l2)
        
        results = {}
        
        # ===== Evaluate FAISS Baseline =====
        print("\nEvaluating FAISS baseline (ground truth ceiling)...")
        faiss_recalls = []
        faiss_times = []
        
        for query_emb, gt in zip(dataset["query_embeddings"], dataset["ground_truth"]):
            faiss_results, search_time = index.search_faiss(query_emb, k)
            recall = len(set(faiss_results) & set(gt)) / len(gt) if gt else 0
            faiss_recalls.append(recall)
            faiss_times.append(search_time)
        
        results["FAISS (flat) [baseline]"] = {
            "recall": np.mean(faiss_recalls),
            "recall_std": np.std(faiss_recalls),
            "time_ms": np.mean(faiss_times) * 1000,
            "buckets": total_buckets,
            "buckets_pct": 100.0,
            "teleports": 0,
            "order": -1  # Always first
        }
        
        # ===== Evaluate HNQT Configurations =====
        for order, config in enumerate(search_configs):
            print(f"Evaluating {config['name']}...")
            
            recalls = []
            times = []
            buckets_list = []
            teleports_list = []
            
            for query_emb, gt in zip(dataset["query_embeddings"], dataset["ground_truth"]):
                hnqt_results, stats = index.search(
                    query_emb, k,
                    n_probe_l1=config["n_probe_l1"],
                    n_probe_l2=config["n_probe_l2"],
                    use_teleport=config["use_teleport"]
                )
                
                recall = len(set(hnqt_results) & set(gt)) / len(gt) if gt else 0
                recalls.append(recall)
                times.append(stats["search_time"])
                buckets_list.append(stats["buckets_searched"])
                teleports_list.append(stats["teleports_used"])
            
            avg_buckets = np.mean(buckets_list)
            results[config["name"]] = {
                "recall": np.mean(recalls),
                "recall_std": np.std(recalls),
                "time_ms": np.mean(times) * 1000,
                "buckets": avg_buckets,
                "buckets_pct": (avg_buckets / total_buckets) * 100,
                "teleports": np.mean(teleports_list),
                "order": order
            }
        
        return results
    
    def print_results(self, results: Dict[str, Any], k: int = 10):
        """Pretty print evaluation results sorted by aggressiveness (recall ascending)"""
        print("\n" + "=" * 100)
        print(f"EVALUATION RESULTS (Recall@{k}) - Sorted by Search Aggressiveness")
        print("=" * 100)
        
        # Header
        print(f"\n{'Configuration':<35} {'Recall':<16} {'Time (ms)':<12} {'Buckets':<12} {'% Index':<10} {'Teleports':<10}")
        print("-" * 100)
        
        # Sort by order (least to most aggressive), then recall
        sorted_results = sorted(results.items(), key=lambda x: (x[1]["order"], x[1]["recall"]))
        
        # Separate baseline
        baseline = None
        hnqt_results = []
        for name, metrics in sorted_results:
            if "baseline" in name:
                baseline = (name, metrics)
            else:
                hnqt_results.append((name, metrics))
        
        # Print HNQT results first (least to most aggressive)
        prev_recall = 0
        for name, metrics in hnqt_results:
            recall_str = f"{metrics['recall']:.3f} (±{metrics['recall_std']:.3f})"
            time_str = f"{metrics['time_ms']:.2f}"
            buckets_str = f"{metrics['buckets']:.1f}"
            pct_str = f"{metrics['buckets_pct']:.1f}%"
            teleports_str = f"{metrics['teleports']:.1f}"
            
            # Add indicator for recall improvement
            improvement = ""
            if metrics['recall'] > prev_recall + 0.01:
                improvement = " ↑"
            prev_recall = metrics['recall']
            
            print(f"{name:<35} {recall_str:<16} {time_str:<12} {buckets_str:<12} {pct_str:<10} {teleports_str:<10}{improvement}")
        
        # Print baseline last
        print("-" * 100)
        if baseline:
            name, metrics = baseline
            recall_str = f"{metrics['recall']:.3f} (±{metrics['recall_std']:.3f})"
            time_str = f"{metrics['time_ms']:.2f}"
            buckets_str = f"{metrics['buckets']:.0f}"
            pct_str = "100.0%"
            teleports_str = "N/A"
            print(f"{name:<35} {recall_str:<16} {time_str:<12} {buckets_str:<12} {pct_str:<10} {teleports_str:<10}")
        
        # Summary statistics
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        
        if baseline:
            baseline_recall = baseline[1]["recall"]
            
            # Find key thresholds
            for threshold in [0.90, 0.95, 0.99]:
                for name, metrics in hnqt_results:
                    if metrics["recall"] >= threshold * baseline_recall:
                        print(f"  {threshold*100:.0f}% of baseline recall: {name} "
                              f"(recall={metrics['recall']:.3f}, {metrics['buckets_pct']:.1f}% of index)")
                        break
            
            # Best efficiency (highest recall per bucket)
            efficiencies = [(name, m["recall"] / max(m["buckets"], 1)) 
                           for name, m in hnqt_results if m["buckets"] > 0]
            if efficiencies:
                best_eff = max(efficiencies, key=lambda x: x[1])
                best_metrics = results[best_eff[0]]
                print(f"\n  Most efficient: {best_eff[0]} "
                      f"(recall={best_metrics['recall']:.3f}, {best_metrics['buckets_pct']:.1f}% of index)")
            
            # Best recall
            best_hnqt = max(hnqt_results, key=lambda x: x[1]["recall"])
            print(f"  Best recall: {best_hnqt[0]} "
                  f"(recall={best_hnqt[1]['recall']:.3f}, {best_hnqt[1]['buckets_pct']:.1f}% of index)")
            print(f"\n  HNQT achieves {best_hnqt[1]['recall']/baseline_recall*100:.1f}% of baseline recall")
    
    def diagnose_query(self, index: HNQTIndex, dataset: Dict[str, Any], query_idx: int = 0):
        """Diagnose where recall is lost for a specific query"""
        query = dataset["queries"][query_idx]
        query_emb = dataset["query_embeddings"][query_idx]
        gt_indices = dataset["ground_truth"][query_idx]
        
        print(f"\n{'='*70}")
        print(f"DIAGNOSIS: Query {query_idx}")
        print(f"{'='*70}")
        print(f"Query: {query[:70]}...")
        print(f"Ground truth: {gt_indices[:5]}... ({len(gt_indices)} total)")
        
        # Where does query route?
        l1_code, l2_code, l1_idx, l2_idx = index.quantizer.quantize(query_emb)
        print(f"\nQuery routes to: {l1_code}/{l2_code}")
        
        # Where are ground truth docs?
        gt_buckets = defaultdict(list)
        for gt_idx in gt_indices:
            doc_emb = dataset["doc_embeddings"][gt_idx]
            l1_c, l2_c, _, _ = index.quantizer.quantize(doc_emb)
            gt_buckets[f"{l1_c}/{l2_c}"].append(gt_idx)
        
        print(f"\nGround truth distribution:")
        for bucket, indices in sorted(gt_buckets.items()):
            match = "✓" if bucket == f"{l1_code}/{l2_code}" else " "
            print(f"  {match} {bucket}: {len(indices)} docs {indices[:3]}...")
        
        # Show cross-links from query bucket
        cross_links = index.quantizer.get_cross_links(l1_idx, l2_idx)
        if cross_links:
            print(f"\nCross-links from {l1_code}/{l2_code}:")
            for link in cross_links:
                target_l1_code = index.quantizer.l1_codes[link.target_l1]
                target_l2_code = index.quantizer.l2_codes[link.target_l1][link.target_l2]
                target_bucket = f"{target_l1_code}/{target_l2_code}"
                has_gt = "✓" if target_bucket in gt_buckets else " "
                print(f"  {has_gt} → {target_bucket} (sim={link.similarity:.3f})")


# ============================================================================
# MAIN
# ============================================================================

def main(dataset_name: str = "synthetic", max_docs: int = 1000, max_queries: int = 50):
    """
    Run the complete HNQT v7 evaluation
    
    Args:
        dataset_name: 'synthetic' or 'bioasq'
        max_docs: Maximum documents to use
        max_queries: Maximum queries to use
    """
    print("=" * 100)
    print("HNQT v7: Comprehensive Evaluation with Aggressive Search Configs")
    print("=" * 100)
    
    # Create evaluator
    evaluator = HNQTEvaluator()
    
    # Load dataset
    if dataset_name == "synthetic":
        print("\nUsing SYNTHETIC dataset")
        dataset = evaluator.create_synthetic_dataset(n_docs=max_docs, n_queries=max_queries)
    elif dataset_name == "bioasq":
        print(f"\nUsing BioASQ dataset")
        dataset = evaluator.load_bioasq(max_docs=max_docs, max_queries=max_queries)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'synthetic' or 'bioasq'")
    
    # Determine cluster sizes based on dataset size
    n_docs_actual = len(dataset["documents"])
    
    # Heuristic: sqrt(n) clusters at L1, sqrt(n/L1) at L2
    n_l1 = max(2, min(8, int(np.sqrt(n_docs_actual / 16))))
    n_l2 = max(2, min(8, int(np.sqrt(n_docs_actual / n_l1 / 4))))
    
    print(f"\nDataset: {dataset['name']}")
    print(f"  Documents: {n_docs_actual}")
    print(f"  Queries: {len(dataset['queries'])}")
    print(f"  Auto-selected clusters: L1={n_l1}, L2={n_l2} ({n_l1 * n_l2} total buckets)")
    
    # Build HNQT index
    print("\n" + "=" * 100)
    print("BUILDING INDEX")
    print("=" * 100)
    
    index = HNQTIndex(
        base_path="vdb_v7",
        n_clusters_l1=n_l1,
        n_clusters_l2=n_l2
    )
    index.build(dataset["documents"], dataset["doc_embeddings"])
    
    # Run evaluation
    print("\n" + "=" * 100)
    print("RUNNING EVALUATION")
    print("=" * 100)
    
    results = evaluator.evaluate(index, dataset, k=10)
    evaluator.print_results(results, k=10)
    
    # Diagnose a few queries
    print("\n" + "=" * 100)
    print("QUERY DIAGNOSTICS (Sample)")
    print("=" * 100)
    
    n_queries = len(dataset["queries"])
    diagnostic_indices = [0, n_queries // 3, 2 * n_queries // 3]
    
    for i in diagnostic_indices:
        if i < n_queries:
            evaluator.diagnose_query(index, dataset, query_idx=i)
    
    return index, results, dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HNQT v7 Comprehensive Evaluation")
    parser.add_argument("--dataset", default="synthetic",
                       choices=["synthetic", "bioasq"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--max-docs", type=int, default=1000,
                       help="Maximum documents to index")
    parser.add_argument("--max-queries", type=int, default=50,
                       help="Maximum queries to evaluate")
    
    args = parser.parse_args()
    
    index, results, dataset = main(
        dataset_name=args.dataset,
        max_docs=args.max_docs,
        max_queries=args.max_queries
    )