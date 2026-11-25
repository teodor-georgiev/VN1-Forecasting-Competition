import numpy as np
import pytest
import vn1_residual_regime_matrix as v
from test_vn1_persistent_bias_specialist import history

def test_conserve_eligible_pool_and_zero_fallback():
    p = np.full((4,13), 10.)
    mask = np.array([True, True, False, False])
    q = p.copy()
    q[0] *= 2
    out = v._conserve(q,p,mask)
    np.testing.assert_allclose(out.sum(axis=0),p.sum(axis=0))
    np.testing.assert_array_equal(out[~mask],p[~mask])
    assert out[0,0]>p[0,0]
    np.testing.assert_array_equal(v._conserve(np.zeros_like(p),p,mask),p)

def test_intermittent_probability_and_dormant_fallback():
    y = np.zeros((3, 131))
    y[0,-26::4] = 8.
    y[1,-26:] = 5.
    assert v._regimes(y)[0] == 3
    assert v._regimes(y)[2] == 4
    expert = v._experts(y)[2]
    assert 0 < expert[0,0] < 8.
    assert expert[2].sum() == 0

def test_future_records_rejected_before_model_fit():
    h = history(40,131)
    p = np.ones((40,13))
    with pytest.raises(ValueError,match="chronology"):
        v.candidate_forecasts(h,p,[{"origin":119}])

def test_completed_oof_matrix_preserves_inactive_and_total(monkeypatch):
    h = history(40,131)
    y = v.base._history_matrix(h)
    p = np.full((40,13),10.)
    records = [{"origin":118,"champion":p,"truth":y[:,118:131]}]
    monkeypatch.setattr(v.base,"_regression_features",lambda *a: None)
    monkeypatch.setattr(v,"_residual_models",lambda *a: (p.copy(),-p.copy()))
    out = v.candidate_forecasts(h,p,records,original_champion=p)
    assert tuple(out) == v.POLICIES
    for q in out.values():
        np.testing.assert_allclose(q.sum(axis=0),p.sum(axis=0))
        np.testing.assert_array_equal(q[-1],p[-1])
        assert (q>=0).all()
