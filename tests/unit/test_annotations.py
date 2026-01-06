import tempfile, csv
from pathbench.core.annotations import ClassificationAnnotation, RegressionAnnotation, SurvivalAnnotation




def _write(tmp, header, rows):
    p = tmp / "ann.csv"
    with open(p, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
    return str(p)




def test_classification_schema(tmp_path):
    p = _write(tmp_path,
    ["slide","patient","dataset","category"],
    [["s1.svs","p1","d1",0]])
    rows = ClassificationAnnotation().read(p)
    assert rows[0].payload["category"] == 0




def test_continuous_survival_schema(tmp_path):
    p = _write(tmp_path,
    ["slide","patient","dataset","time","event"],
    [["s1","p1","d1",26,1]])
    rows = SurvivalAnnotation().read(p)
    assert rows[0].payload["event"] in (0,1)
    
def test_discrete_survival_schema(tmp_path):
    p = _write(tmp_path,
    ["slide","patient","dataset","time_bin","event"],
    [["s1","p1","d1",2,0]])
    rows = SurvivalAnnotation(discrete=True).read(p)
    assert rows[0].payload["time_bin"] == 2
    assert rows[0].payload["event"] in (0,1)

def test_regression_schema(tmp_path):
    p = _write(tmp_path,
    ["slide","patient","dataset","value"],
    [["s1","p1","d1",3.14]])
    rows = RegressionAnnotation().read(p)
    assert abs(rows[0].payload["value"] - 3.14) < 1e-6