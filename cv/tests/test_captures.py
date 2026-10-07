import pytest

from facade_cv.io.captures import load_captures

HEADER = "file,building,type,date,weather,height_m,note"


def _setup(tmp_path, lines):
    for rel in ("b01_a/x_01.MOV", "b01_a/x_02.mov", "b02_b/y_01.MOV"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).touch()
    (tmp_path / "captures.csv").write_bytes(("\r\n".join(lines)).encode("utf-8-sig"))
    return tmp_path


def test_excel_wrapped_rows(tmp_path):
    """실제 captures.csv 형식: BOM + CRLF + 행 전체가 따옴표로 감싸짐, 파일명 대소문자 불일치."""
    d = _setup(tmp_path, [f'"{HEADER}"', '"x_02.mov,b01,ground,2026-10-07,sunny,,정지"',
                          '"x_01.mov,b01,close,2026-10-07,sunny,,"', '"y_01.mov,b02,ground,2026-10-07,sunny,12,"'])
    caps = load_captures(d)
    assert list(caps) == ["b01", "b02"]
    assert [r["file"] for r in caps["b01"]] == ["x_01.mov", "x_02.mov"]
    assert caps["b01"][0]["path"] == d / "b01_a" / "x_01.MOV"
    assert caps["b01"][1]["note"] == "정지"
    assert caps["b02"][0]["height_m"] == "12"


def test_plain_csv_with_quoted_field(tmp_path):
    d = _setup(tmp_path, [HEADER, 'y_01.mov,b02,ground,2026-10-07,sunny,,"반사, 오염"'])
    assert load_captures(d)["b02"][0]["note"] == "반사, 오염"


def test_missing_video(tmp_path):
    d = _setup(tmp_path, [HEADER, "nope.mov,b09,ground,2026-10-07,sunny,,"])
    with pytest.raises(FileNotFoundError, match="nope.mov"):
        load_captures(d)
