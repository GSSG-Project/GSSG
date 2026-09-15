import os

import torch

from gssg.scene_graph.vector_db import FaissHNSWVectorDB


def print_header(msg):
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def run_tests():
    dim = 128
    save_prefix = "my_test_db"

    for ext in [".index", ".pkl"]:
        if os.path.exists(save_prefix + ext):
            os.remove(save_prefix + ext)

    db = FaissHNSWVectorDB(dim=dim, device="cpu")

    print_header("1. BASIC INSERT & SEARCH")
    vec_a = torch.randn(dim)
    vec_b = torch.randn(dim)
    vec_c = torch.randn(dim)

    id_a = db.insert(vec_a, metadata={"object_id": 123})
    id_b = db.insert(vec_b, metadata={"object_id": 124})
    id_c = db.insert(vec_c, metadata={"object_id": 125})

    assert len(db) == 3, f"Expected DB size 3, but got {len(db)}"
    print(f"Inserted 3 vectors. DB size: {len(db)}")

    results = db.search(vec_a, k=1)
    print(f"Search results for vec_a: {results}")
    assert results[0][0] == id_a
    assert results[0][1] > 0.999

    print_header("2. VECTOR & METADATA RETRIEVAL")
    retrieved_vec_a = db.get_vector(id_a)
    retrieved_meta_b = db.get_metadata(id_b)

    assert isinstance(retrieved_vec_a, torch.Tensor), "Retrieved vector should be a Torch Tensor"
    assert torch.allclose(retrieved_vec_a, vec_a, atol=1e-6), (
        "Retrieved vector differs from original"
    )
    print("SUCCESS: Vector retrieval works and returns the original torch.Tensor.")

    assert retrieved_meta_b["object_id"] == 124, "Retrieved metadata is incorrect"
    print("SUCCESS: Metadata retrieval works.")

    print_header("3. DEDUPLICATION")
    vec_a_duplicate = vec_a + (torch.randn(dim) * 0.0001)
    id_dup = db.insert(vec_a_duplicate, similarity_threshold=0.999)

    print(f"Original ID: {id_a}, Returned ID for near-duplicate: {id_dup}")
    assert id_a == id_dup
    assert len(db) == 3
    print("SUCCESS: Deduplication prevented new entry.")

    print_header("4. DELETION")
    db.delete(id_b)
    print(f"Deleted vector with ID {id_b}. Current DB size: {len(db)}")
    assert len(db) == 2
    assert db.get_vector(id_b) is None

    results_after_delete = db.search(vec_a, k=1)
    assert results_after_delete[0][0] == id_a

    results_for_deleted = db.search(vec_b, k=1)
    if results_for_deleted:
        assert results_for_deleted[0][0] != id_b
    print("SUCCESS: Deletion works as expected.")

    print_header("5. PERSISTENCE (SAVE / LOAD)")
    db.save(save_prefix)
    print("Database saved.")

    new_db = FaissHNSWVectorDB(dim=dim)
    new_db.load(save_prefix)
    print("New DB instance loaded from files.")

    assert len(new_db) == 2
    assert new_db.get_metadata(id_a)["object_id"] == 123
    loaded_results = new_db.search(vec_c, k=1)
    assert loaded_results[0][0] == id_c
    print("SUCCESS: Persistence and loading are correct.")

    os.remove(f"{save_prefix}.index")
    os.remove(f"{save_prefix}.pkl")
    print("\nCleanup: Test files deleted.")

    print("\n" + "=" * 20 + " ALL TESTS PASSED SUCCESSFULLY! " + "=" * 20)


if __name__ == "__main__":
    run_tests()
