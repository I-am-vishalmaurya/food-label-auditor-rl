# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest configuration -- ensures module1/ is on sys.path."""

import sys
from pathlib import Path

MODULE1_DIR = Path(__file__).resolve().parent.parent.parent
if str(MODULE1_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE1_DIR))
