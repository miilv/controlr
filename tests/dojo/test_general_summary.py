"""Preregistered all-cell regression gates and layout clustering, without GPU/data."""
import importlib.util
from pathlib import Path

p=Path(__file__).resolve().parents[2]/'scripts/perception/sam/general_summary.py'
spec=importlib.util.spec_from_file_location('general_summary',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(iou=.5, phantoms=0, key='same-input'):
    return {'task':'cell','layout_sha256':key,'rows':[{'px':40,'clutter':False,'label':'object','iou':iou}], 'phantoms':phantoms}


def test_phantom_harm_needs_rate_and_count():
    base=m.aggregate([row(),row(),row(),row(),row()])
    small=m.aggregate([row(phantoms=1),row(),row(),row(),row()])
    large=m.aggregate([row(phantoms=3),row(),row(),row(),row()])
    assert not m.harms(base,small)
    assert 'phantoms_increase>=0.5/image_and_2_total' in m.harms(base,large)


def test_recall_and_iou_harm_are_not_waived_by_extra_masks():
    base=m.aggregate([row(.6)])
    wrong=m.aggregate([row(.49)])
    assert {'recall_drop>=5pp','IoU_drop>=0.02'} <= set(m.harms(base,wrong))


def test_camera_views_of_one_input_are_one_uncertainty_cluster():
    a=[row(.49),row(.49),row(.49)]
    b=[row(.51),row(.51),row(.51)]
    ci=m.paired_ci(a,b)
    assert ci['clusters']==1
    assert ci['recall_delta_95ci']==[1.,1.]


def test_no_visible_objects_still_has_phantom_guard():
    a=row();a['rows']=[]
    b={**a,'phantoms':2}
    assert m.aggregate([a])['recall'] is None
    assert m.harms(m.aggregate([a]),m.aggregate([b]))
