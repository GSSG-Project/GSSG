import logging
import os
import pickle

import faiss
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Cosine similarity at/above which an inserted vector is treated as a duplicate
# of an existing one (returns the existing id instead of adding a new vector).
DEDUP_SIMILARITY_THRESHOLD = 0.999


class FaissHNSWVectorDB:
    def __init__(
        self,
        dim: int,
        M: int = 32,
        ef_search: int = 64,
        ef_construction: int = 200,
        device: str = "cpu",
    ):
        self.dim = dim
        self.device = device

        # Save config to rebuild index later
        self.M = M
        self.ef_search = ef_search
        self.ef_construction = ef_construction

        self._init_index()

        self.deleted_ids = set()
        self.next_id = 0
        self.id_to_metadata: dict[int, dict] = {}
        self.id_to_vector: dict[int, torch.Tensor] = {}

    def _init_index(self):
        index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        self.index_id_map = faiss.IndexIDMap(index)

    def __len__(self) -> int:
        return len(self.id_to_vector)

    def _normalize_vector(self, vector: torch.Tensor) -> np.ndarray:
        if vector.dim() == 2 and vector.shape[0] == 1:
            vector = vector.squeeze(0)

        if vector.dim() != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Input vector must be 1D with dimension {self.dim}")

        normalized_vector = torch.nn.functional.normalize(vector.to(self.device), p=2, dim=0)
        return normalized_vector.cpu().numpy().astype("float32").reshape(1, -1)

    def insert(
        self,
        vector: torch.Tensor,
        metadata: dict | None = None,
        vec_id: int | None = None,
        similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    ) -> int:
        np_vector = self._normalize_vector(vector)

        if vec_id is not None:
            if vec_id in self.id_to_vector:
                self.delete(vec_id)

        if len(self) > 0:
            search_k = min(self.index_id_map.ntotal, 5 + len(self.deleted_ids))
            if search_k > 0:
                distances, ids = self.index_id_map.search(np_vector, k=search_k)
                if ids.size > 0:
                    for i in range(ids.shape[1]):
                        best_id = int(ids[0][i])
                        if best_id != -1 and best_id not in self.deleted_ids:
                            if distances[0][i] >= similarity_threshold:
                                logging.info(
                                    f"[VECTORDB] Very similar vector found (ID: {best_id}). Returning existing ID."
                                )
                                return best_id
                            break

        new_id = self.next_id
        self.index_id_map.add_with_ids(np_vector, np.array([new_id], dtype=np.int64))
        if metadata:
            self.id_to_metadata[new_id] = metadata
        self.id_to_vector[new_id] = vector.cpu()
        self.next_id += 1
        return new_id

    def search(self, query_vector: torch.Tensor, k: int = 5) -> list[tuple[int, float, dict]]:
        if len(self) == 0:
            return []
        np_vector = self._normalize_vector(query_vector)
        search_k = min(self.index_id_map.ntotal, k + len(self.deleted_ids))
        if search_k == 0:
            return []

        distances, ids = self.index_id_map.search(np_vector, search_k)
        results = []
        for i, score in zip(ids[0], distances[0], strict=False):
            if i == -1 or len(results) >= k:
                break
            vec_id = int(i)
            if vec_id not in self.deleted_ids:
                results.append((vec_id, float(score), self.get_metadata(vec_id)))
        return results

    def get_vector(self, vec_id: int) -> torch.Tensor | None:
        vector = self.id_to_vector.get(vec_id)
        return vector.to(self.device) if vector is not None else None

    def get_metadata(self, vec_id: int) -> dict:
        return self.id_to_metadata.get(vec_id, {})

    def delete(self, vec_id: int):
        if vec_id in self.id_to_vector:
            self.deleted_ids.add(vec_id)
            self.id_to_metadata.pop(vec_id, None)
            self.id_to_vector.pop(vec_id, None)
            logging.info(f"[VECTORDB] Deleted ID: {vec_id}")
        else:
            logging.warning(f"Attempted to delete non-existent ID: {vec_id}")

    def consolidate_index(self):
        if not self.deleted_ids:
            return
        logging.info("Consolidating index... (Removing deleted vectors and re-indexing)")

        self._init_index()

        if not self.id_to_vector:
            self.deleted_ids = set()
            return

        ids_list = []
        vectors_list = []

        for vec_id, vec_tensor in self.id_to_vector.items():
            norm_vec = self._normalize_vector(vec_tensor)
            vectors_list.append(norm_vec)
            ids_list.append(vec_id)

        if vectors_list:
            vectors_np = np.vstack(vectors_list)
            ids_np = np.array(ids_list, dtype=np.int64)

            self.index_id_map.add_with_ids(vectors_np, ids_np)

        self.deleted_ids = set()
        logging.info(f"[VECTORDB] Index consolidated. Total items: {self.index_id_map.ntotal}")

    def get_most_similar_vector_from_list(self, vector_id, vector_id_list):
        source_vector = self.get_vector(vector_id)
        source_vector = source_vector.unsqueeze(0)

        best_match_id = None
        best_similarity = -1.0

        for candidate_id in vector_id_list:
            candidate_vector = self.get_vector(candidate_id)
            candidate_vector = candidate_vector.unsqueeze(0)

            similarity = F.cosine_similarity(source_vector, candidate_vector).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = candidate_id

        return best_match_id, self.get_metadata(best_match_id)

    def save(self, filepath_prefix: str, consolidate: bool = True):
        if consolidate:
            self.consolidate_index()

        output_dir = os.path.dirname(filepath_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        faiss.write_index(self.index_id_map, f"{filepath_prefix}.index")

        data_to_save = {
            "dim": self.dim,
            "next_id": self.next_id,
            "deleted_ids": self.deleted_ids,
            "id_to_metadata": self.id_to_metadata,
            "id_to_vector": self.id_to_vector,
            "config": {
                "M": self.M,
                "ef_search": self.ef_search,
                "ef_construction": self.ef_construction,
            },
        }
        with open(f"{filepath_prefix}.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
        logging.info(f"[VECTORDB] Database saved to {filepath_prefix}.index and .pkl")

    def load(self, filepath_prefix: str):
        index_path = f"{filepath_prefix}.index"
        pkl_path = f"{filepath_prefix}.pkl"
        if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
            raise FileNotFoundError(f"Database files not found at prefix: {filepath_prefix}")

        self.index_id_map = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.dim = data["dim"]
        self.next_id = data["next_id"]
        self.deleted_ids = data["deleted_ids"]
        self.id_to_metadata = data["id_to_metadata"]
        self.id_to_vector = data["id_to_vector"]

        if "config" in data:
            self.M = data["config"]["M"]
            self.ef_search = data["config"]["ef_search"]
            self.ef_construction = data["config"]["ef_construction"]

        logging.info(f"[VECTORDB] Database loaded successfully from {filepath_prefix}")
