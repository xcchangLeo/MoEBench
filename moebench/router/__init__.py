"""Router modeling for selecting UnixBench expert subsets."""

from moebench.router.feature_vectorizer import XiVectorizer
from moebench.router.dataset_loader import load_unixbench_dataset_for_router, RouterDataset

__all__ = [
    "XiVectorizer",
    "load_unixbench_dataset_for_router",
    "RouterDataset",
    "predict_expert_scores",
    "select_top_k_from_probs",
]


def __getattr__(name: str):
    if name == "predict_expert_scores":
        from moebench.router.inference import predict_expert_scores

        return predict_expert_scores
    if name == "select_top_k_from_probs":
        from moebench.router.inference import select_top_k_from_probs

        return select_top_k_from_probs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

