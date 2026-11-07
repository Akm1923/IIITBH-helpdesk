import pytest
from openpyxl import Workbook

from app.application.evaluation.qa_dataset import DEFAULT_DATASET_PATH, load_qa_pairs


def test_load_qa_pairs_reads_pair_question_and_answer(tmp_path) -> None:
    path = tmp_path / "pairs.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Pair ID", "Question", "Answer"])
    sheet.append([1, "What is the course code?", "CS203"])
    sheet.append([None, None, None])
    workbook.save(path)

    pairs = load_qa_pairs(path)

    assert len(pairs) == 1
    assert pairs[0].pair_id == "1"
    assert pairs[0].question == "What is the course code?"
    assert pairs[0].answer == "CS203"
    assert pairs[0].dataset_item_id == "iiitbh-qa-1"


def test_bundled_workbook_has_two_hundred_pairs() -> None:
    pairs = load_qa_pairs(DEFAULT_DATASET_PATH)

    assert len(pairs) == 200
    assert pairs[0].question
    assert pairs[0].answer


@pytest.mark.parametrize("ids", [[None], ["  "], [1, " 1 "], [1.0, "1"]])
def test_load_rejects_missing_or_duplicate_ids(tmp_path, ids):
    path = tmp_path / "invalid.xlsx"
    workbook = Workbook()
    workbook.active.append(["Pair ID", "Question", "Answer"])
    for pair_id in ids:
        workbook.active.append([pair_id, "Question?", "Answer"])
    workbook.save(path)
    workbook.close()
    with pytest.raises(ValueError, match="Pair ID"):
        load_qa_pairs(path)
