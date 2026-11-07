from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "evals" / "datasets" / "IIITBH_QA_Pairs_200.xlsx"
)
DATASET_NAME = "iiitbh-qa-pairs"


@dataclass(frozen=True)
class QAPair:
    pair_id: str
    question: str
    answer: str

    @property
    def dataset_item_id(self) -> str:
        return "iiitbh-qa-" + self.pair_id


def load_qa_pairs(path: Path | None = None) -> list[QAPair]:
    """Load golden question/answer rows from the evaluation workbook."""
    workbook_path = path or DEFAULT_DATASET_PATH
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            raise ValueError("The Q&A workbook is empty.")
        columns = [str(cell).strip().lower() if cell is not None else "" for cell in header]
        try:
            id_index = columns.index("pair id")
            question_index = columns.index("question")
            answer_index = columns.index("answer")
        except ValueError as exc:
            raise ValueError(
                "The Q&A workbook must have Pair ID, Question and Answer columns."
            ) from exc

        pairs: list[QAPair] = []
        for row_number, row in enumerate(rows, start=2):
            if row is None or all(cell is None or str(cell).strip() == "" for cell in row):
                continue
            question = "" if row[question_index] is None else str(row[question_index]).strip()
            answer = "" if row[answer_index] is None else str(row[answer_index]).strip()
            if not question or not answer:
                raise ValueError(f"Row {row_number} is missing a question or answer.")
            raw_id = row[id_index]
            pair_id = str(int(raw_id)) if isinstance(raw_id, float) and raw_id.is_integer() else str(raw_id).strip()
            pairs.append(QAPair(pair_id=pair_id, question=question, answer=answer))
        if not pairs:
            raise ValueError("The Q&A workbook has no question and answer rows.")
        return pairs
    finally:
        workbook.close()
