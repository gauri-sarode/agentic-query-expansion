from unittest.mock import patch

from src.observability.embeddings import cosine_similarity, mean_pairwise_similarity


def test_cosine_similarity_identical_vectors():
    v = (1.0, 2.0, 3.0)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors():
    assert abs(cosine_similarity((1.0, 0.0), (0.0, 1.0))) < 1e-9


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert cosine_similarity((0.0, 0.0), (1.0, 2.0)) == 0.0


def test_mean_pairwise_similarity_single_text_is_one():
    assert mean_pairwise_similarity(["only one"]) == 1.0


def test_mean_pairwise_similarity_averages_pairs():
    fake_vecs = {
        "a": (1.0, 0.0),
        "b": (1.0, 0.0),  # identical to a -> sim 1.0
        "c": (0.0, 1.0),  # orthogonal to both -> sim 0.0
    }
    with patch("src.observability.embeddings.embed", side_effect=lambda t: fake_vecs[t]):
        result = mean_pairwise_similarity(["a", "b", "c"])
    # pairs: (a,b)=1.0, (a,c)=0.0, (b,c)=0.0 -> mean = 1/3
    assert abs(result - (1.0 / 3)) < 1e-9
