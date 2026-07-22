"""
BayeSED3 module entry: ``python -m bayesed`` → Launcher (ADR-0007).

For the Python Params / Inference path, use ``BayeSEDInterface`` /
``SEDInference`` instead of this module.
"""

from .launcher import main

if __name__ == "__main__":
    main()
