"""Bölünme ve sızıntı kontrolleri."""
from src.data import assign_splits, build_manifest, check_no_leakage, parse_filename
from pathlib import Path


def test_parse_filename():
    m = parse_filename(Path("x/fan/train/normal_id_02_00000123.wav"))
    assert m == {"machine_type": "fan", "machine_id": "id_02", "label": "normal",
                 "official_split": "train", "source_recording_id": "train/normal_id_02_00000123"}
    assert parse_filename(Path("x/fan/test/anomaly_id_06_00000001.wav"))["label"] == "anomaly"
    assert parse_filename(Path("x/fan/other/normal_id_00_00000000.wav")) is None
    assert parse_filename(Path("x/fan/train/readme.wav")) is None


def test_manifest_and_split(mini_dataset):
    cfg = mini_dataset
    man = build_manifest(cfg, progress=False)
    assert set(man["machine_id"]) == {"id_00", "id_04"}
    short = man[man["path"].str.endswith("00000052.wav")]
    assert not short["usable"].iloc[0] and "too_short" in short["issue"].iloc[0]

    df = assign_splits(man, cfg)
    assert check_no_leakage(df) == []
    # Resmî testin tamamı testte, başka hiçbir şey testte değil
    assert (df[df.official_split == "test"]["split"] == "test").all()
    assert set(df[df.split == "test"]["official_split"]) == {"test"}
    # Anomali yalnızca testte
    assert set(df[df.label == "anomaly"]["split"]) == {"test"}
    # Kopya çifti aynı bölümde
    a = df[df.path.str.endswith("train/normal_id_00_00000000.wav")]["split"].iloc[0]
    b = df[df.path.str.endswith("train/normal_id_00_00000050.wav")]["split"].iloc[0]
    assert a == b
    # Test kaydının train'deki kopyası dışlandı
    assert df[df.path.str.endswith("train/normal_id_00_00000051.wav")]["split"].iloc[0] == "excluded_dup_of_test"
    # Oranlar yaklaşık %70/%15/%15
    sub = df[(df.machine_id == "id_04") & df.split.isin(["train", "val", "cal"])]
    counts = sub["split"].value_counts()
    assert counts["train"] == 14 and counts["val"] == 3 and counts["cal"] == 3


def test_split_deterministic(mini_dataset):
    cfg = mini_dataset
    man = build_manifest(cfg, progress=False)
    s1 = assign_splits(man, cfg)["split"].tolist()
    s2 = assign_splits(man, cfg)["split"].tolist()
    assert s1 == s2
    cfg2 = {**cfg, "seed": 7}
    assert assign_splits(man, cfg2)["split"].tolist() != s1


def test_leakage_detected(mini_dataset):
    cfg = mini_dataset
    df = assign_splits(build_manifest(cfg, progress=False), cfg)
    i = df.index[(df.label == "anomaly")][0]
    df.loc[i, "split"] = "train"
    assert any("anomali" in p for p in check_no_leakage(df))
