from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evals.run_rag_experiment import sync_dataset


@pytest.mark.parametrize("status", [None, 500, 409])
def test_sync_only_ignores_confirmed_conflict(status):
    error = RuntimeError("already exists 409")
    if status is not None:
        error.status_code = status
    client = Mock()
    client.create_dataset.side_effect = error
    pair = SimpleNamespace(dataset_item_id="id", question="q", answer="a", pair_id="1")
    if status == 409:
        sync_dataset(client, [pair])
        client.create_dataset_item.assert_called_once()
    else:
        with pytest.raises(RuntimeError):
            sync_dataset(client, [pair])
        client.create_dataset_item.assert_not_called()
