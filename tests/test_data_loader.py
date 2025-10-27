from pathlib import Path
from hw5lib.data import load_split, DataLoader

def test_loader_roundtrip(tmp_path: Path):
    p = tmp_path / "fake.csv"
    p.write_text(
        "set,feature,target\n"
        "train,1,0\n"
        "train,2,1\n"
        "test,3,0\n"
    )
    tr, te = load_split(p)
    assert (len(tr), len(te)) == (2, 1)

    dl = DataLoader(p)
    tr2, te2 = dl.load()
    assert (len(tr2), len(te2)) == (2, 1)
