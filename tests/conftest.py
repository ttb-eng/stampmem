# -*- coding: utf-8 -*-
"""pytest 公共配置：把项目根加入 sys.path，并提供落在项目内的临时库目录。

为什么不用 pytest 的 tmp_path：本环境下系统临时目录可能被沙箱拦住
（sqlite3.OperationalError: unable to open database file），Chroma 建库会失败。
所以统一建在项目内的 `_pytest_tmp/` 下，用完即删。
"""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_dir(request):
    """给每个测试一个独立的、位于项目内的持久化目录。"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in request.node.name)
    d = ROOT / "_pytest_tmp" / safe
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)
