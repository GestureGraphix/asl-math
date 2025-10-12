"""ASL Translator Package.

Mathematical Linguistics and Scalable Modeling for Large Vocabulary ASL Translation.
"""

__version__ = "0.1.0"
__author__ = "ML/NLP Engineer"

from .normalize import Sim3Normalizer
from .features import ProductVQ, FeatureExtractor
from .spatial import SpatialDiscourse
from .model import CausalTCNEncoder, CTCHead
from .vocab import Vocabulary

__all__ = [
    "Sim3Normalizer",
    "ProductVQ", 
    "FeatureExtractor",
    "SpatialDiscourse",
    "CausalTCNEncoder",
    "CTCHead",
    "Vocabulary",
]