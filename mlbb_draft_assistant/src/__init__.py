"""MLBB Draft Assistant source package.

Intentionally left without imports so that lightweight modules
(``data_manager``, ``scoring_engine``) can be used without importing the heavy
optional dependencies (OpenCV, mss, PyQt5) pulled in by ``vision`` / ``overlay``.
"""
