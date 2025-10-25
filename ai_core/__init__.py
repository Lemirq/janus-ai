"""
Janus AI - Real-time Persuasion Assistant

A sophisticated AI system that provides real-time persuasion assistance
through audio analysis, response generation, and prosody-enhanced speech.
"""

__version__ = "1.0.0"
__author__ = "Janus AI Team"

from .main import JanusAI, JanusConfig
from .core.persuasion_engine import PersuasionObjective

__all__ = ["JanusAI", "JanusConfig", "PersuasionObjective"]
