# SPDX-License-Identifier: Apache-2.0
"""mini-llmcache: a minimal, pedagogical LMCache reimplementation.

A standalone KV-cache server for vLLM: hashed prompt chunks are stored in
L1 (host memory) / L2 (pluggable adapters) and matching prefixes are
replayed to skip prefill.  Works on NVIDIA GPUs and Ascend NPUs.

This package is intentionally lightweight — importing it does not pull in
torch, zmq, or vLLM.
"""

__version__ = "1.0.0"
