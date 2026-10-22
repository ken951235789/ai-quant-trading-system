"""公開研究快照的完整性與聚合一致性；不是策略獲利測試。"""

import hashlib
import json
from pathlib import Path
import re
import statistics

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'docs/research/20261022'


def load(name):
    return json.loads((BUNDLE / name).read_text(encoding='utf-8'))


def test_manifest():
    for name, expected in load('provenance.json')['export_sha256'].items():
        assert Path(name).name == name
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected


def test_catalog_scope():
    data = load('strategy_catalog.json')
    assert len(data['records']) == 156
    assert len({r['id'] for r in data['records']}) == 156
    assert len({r['family_id'] for r in data['records']}) == 20
    assert all(not r['executable'] and not r['live_eligible'] for r in data['records'])
    assert data['summary']['backtests_run'] == 0


def test_feature_scope():
    rows = load('feature_taxonomy.json')['rows']
    assert len(rows) == 210
    assert len({r['group'] for r in rows}) == 41
    assert all(not r['transformer'] for r in rows if r['role'] in {'oos', 'portfolio'})
    assert all(not r['transformer'] and not r['sac'] for r in rows if r['role'] in {'label', 'raw', 'execution'})


@pytest.mark.parametrize('family,base,stress,positive', [
    ('bb_reversion', -3.685, -4.870, 1),
    ('bb_squeeze', -1.501, -1.823, 0),
    ('bb_continuation', -0.599, -1.085, 2),
])
def test_published_account_medians(family, base, stress, positive):
    all_rows = load('rule_factory_accounts.json')['accounts']
    assert len(all_rows) == 456
    rows = [r for r in all_rows if r['path_mode'] == 'rematched'
            and r['window'] == 'Test' and r['family'] == family]
    # 每個成本情境逐一核對，不能把共同進場帳戶混進重新排程結果。
    for scenario, expected in [('base', base), ('stress', stress)]:
        group = [r for r in rows if r['scenario'] == scenario]
        assert len(group) == 15
        assert statistics.median(r['total_return'] for r in group) * 100 == pytest.approx(expected, abs=0.00051)
        assert sum(r['total_return'] > 0 for r in group) == (positive if scenario == 'base' else 0)


def test_release_document_links():
    paths = [ROOT / 'docs/RESEARCH_UPDATE_20261022.md', *BUNDLE.glob('*.md')]
    for path in paths:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#')):
                continue
            assert (path.parent / target.split('#')[0]).is_file(), target
