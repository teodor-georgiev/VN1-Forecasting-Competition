import numpy as np
import pytest
import vn1_calibrated_chronos_regime as v
from test_vn1_persistent_bias_specialist import history

def test_anchor_removes_aggregate_bias_and_zero_fallback():
    c=np.full((4,13),10.)
    f=c*.5
    f[0,0]=20.
    regime=np.array([0,0,1,1])
    out=v._calibrated(c,f,[.30]*4,regime)
    for k in (0,1):
        assert out[regime==k].sum()==pytest.approx(c[regime==k].sum())
    assert out[0,0]>c[0,0]
    np.testing.assert_array_equal(v._calibrated(c,np.zeros_like(f),[.30]*4,regime),c)

def test_selected_policy_and_alignment():
    h=history(40,131)
    target=v.saved._phase2_reference(h)
    target.iloc[:,3:]=10.
    c=np.full((len(h),13),10.)
    f=c*.5
    receipt=v.select(h,target,c,f)
    assert receipt["selected"]=="incumbent"
    assert receipt["phase2_labels_read"] is False
    with pytest.raises(ValueError,match="alignment"):
        v.select(h,target.iloc[::-1].reset_index(drop=True),c,f)

def test_invalid_weight_rejected():
    c=np.ones((2,13))
    with pytest.raises(ValueError,match="weight"):
        v._calibrated(c,c,[.31,0,0,0],np.array([0,0]))
