"""
Setup Pytheus metrics backend based on configuration.
"""

from pytheus.backends import load_backend
from pytheus.metrics import Counter, Histogram


load_backend()
