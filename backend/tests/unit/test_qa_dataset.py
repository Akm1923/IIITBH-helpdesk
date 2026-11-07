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
