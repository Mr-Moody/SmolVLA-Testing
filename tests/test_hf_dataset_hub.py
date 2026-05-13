"""Tests for src/hf/dataset_hub.py — validation, download, and prepare logic."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hf.dataset_hub import (
    DatasetValidationError,
    ValidationResult,
    download_dataset,
    prepare_datasets,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_dataset(tmp_path):
    """Create a minimal valid LeRobotDataset v3 directory."""
    meta = tmp_path / "meta"
    meta.mkdir()
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    videos = tmp_path / "videos" / "observation.images.top" / "chunk-000"
    videos.mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "total_episodes": 25,
        "total_frames": 10000,
        "total_tasks": 5,
        "fps": 30,
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "stats.json").write_text("{}")
    (meta / "tasks.parquet").write_bytes(b"fake-parquet")
    (data / "file-000.parquet").write_bytes(b"fake-data")
    (videos / "file-000.mp4").write_bytes(b"fake-video")

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    """An empty directory — no dataset structure at all."""
    return tmp_path


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateDataset:

    def test_valid_dataset(self, valid_dataset):
        result = validate_dataset(valid_dataset)
        assert result.valid
        assert result.errors == []

    def test_missing_info_json(self, empty_dir):
        result = validate_dataset(empty_dir)
        assert not result.valid
        assert any("meta/info.json not found" in e for e in result.errors)

    def test_invalid_json(self, valid_dataset):
        (valid_dataset / "meta" / "info.json").write_text("{bad json")
        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("invalid JSON" in e for e in result.errors)

    def test_wrong_codebase_version(self, valid_dataset):
        info = json.loads((valid_dataset / "meta" / "info.json").read_text())
        info["codebase_version"] = "v2.0"
        (valid_dataset / "meta" / "info.json").write_text(json.dumps(info))

        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("v2.0" in e for e in result.errors)

    def test_zero_episodes(self, valid_dataset):
        info = json.loads((valid_dataset / "meta" / "info.json").read_text())
        info["total_episodes"] = 0
        (valid_dataset / "meta" / "info.json").write_text(json.dumps(info))

        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("no episodes" in e for e in result.errors)

    def test_zero_frames(self, valid_dataset):
        info = json.loads((valid_dataset / "meta" / "info.json").read_text())
        info["total_frames"] = 0
        (valid_dataset / "meta" / "info.json").write_text(json.dumps(info))

        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("no frames" in e for e in result.errors)

    def test_zero_tasks(self, valid_dataset):
        info = json.loads((valid_dataset / "meta" / "info.json").read_text())
        info["total_tasks"] = 0
        (valid_dataset / "meta" / "info.json").write_text(json.dumps(info))

        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("not annotated" in e for e in result.errors)

    def test_missing_tasks_parquet(self, valid_dataset):
        (valid_dataset / "meta" / "tasks.parquet").unlink()
        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("tasks.parquet not found" in e for e in result.errors)

    def test_missing_data_dir(self, valid_dataset):
        import shutil
        shutil.rmtree(valid_dataset / "data")
        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("data/ directory not found" in e for e in result.errors)

    def test_empty_data_dir(self, valid_dataset):
        (valid_dataset / "data" / "chunk-000" / "file-000.parquet").unlink()
        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("no parquet files" in e for e in result.errors)

    def test_missing_videos_dir(self, valid_dataset):
        import shutil
        shutil.rmtree(valid_dataset / "videos")
        result = validate_dataset(valid_dataset)
        assert not result.valid
        assert any("videos/ directory not found" in e for e in result.errors)

    def test_missing_stats_warning(self, valid_dataset):
        (valid_dataset / "meta" / "stats.json").unlink()
        result = validate_dataset(valid_dataset)
        assert result.valid  # warning, not error
        assert any("stats.json" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Download tests
# ---------------------------------------------------------------------------

class TestDownloadDataset:

    @patch("huggingface_hub.snapshot_download")
    def test_cache_hit_skips_download(self, mock_dl, valid_dataset, tmp_path):
        cache_root = tmp_path / "cache"
        repo_id = "TestUser/test-dataset"
        cached_path = cache_root / repo_id
        cached_path.mkdir(parents=True)
        (cached_path / "meta").mkdir()
        (cached_path / "meta" / "info.json").write_text("{}")

        result = download_dataset(repo_id, cache_root=cache_root)
        assert result == cached_path
        mock_dl.assert_not_called()

    @patch("huggingface_hub.snapshot_download")
    def test_downloads_when_not_cached(self, mock_dl, tmp_path):
        cache_root = tmp_path / "cache"
        repo_id = "TestUser/test-dataset"

        # snapshot_download creates the directory
        def fake_download(**kwargs):
            local_dir = Path(kwargs["local_dir"])
            (local_dir / "meta").mkdir(parents=True, exist_ok=True)
            (local_dir / "meta" / "info.json").write_text("{}")

        mock_dl.side_effect = fake_download

        result = download_dataset(repo_id, cache_root=cache_root)
        assert result == cache_root / repo_id
        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args[1]
        assert call_kwargs["repo_id"] == repo_id
        assert call_kwargs["repo_type"] == "dataset"

    @patch("huggingface_hub.snapshot_download")
    def test_force_redownloads(self, mock_dl, valid_dataset, tmp_path):
        cache_root = tmp_path / "cache"
        repo_id = "TestUser/test-dataset"
        cached_path = cache_root / repo_id
        cached_path.mkdir(parents=True)
        (cached_path / "meta").mkdir()
        (cached_path / "meta" / "info.json").write_text("{}")

        def fake_download(**kwargs):
            local_dir = Path(kwargs["local_dir"])
            (local_dir / "meta").mkdir(parents=True, exist_ok=True)
            (local_dir / "meta" / "info.json").write_text("{}")

        mock_dl.side_effect = fake_download

        download_dataset(repo_id, cache_root=cache_root, force=True)
        mock_dl.assert_called_once()


# ---------------------------------------------------------------------------
# Prepare tests
# ---------------------------------------------------------------------------

class TestPrepareDatasets:

    @patch("hf.dataset_hub.download_dataset")
    @patch("hf.dataset_hub.validate_dataset")
    def test_single_dataset_returns_path(self, mock_val, mock_dl, tmp_path):
        local_path = tmp_path / "ds"
        mock_dl.return_value = local_path
        mock_val.return_value = ValidationResult(valid=True)

        result = prepare_datasets(["TestUser/ds"], cache_root=tmp_path)
        assert result == local_path

    @patch("hf.dataset_hub.download_dataset")
    @patch("hf.dataset_hub.validate_dataset")
    def test_invalid_dataset_raises(self, mock_val, mock_dl, tmp_path):
        mock_dl.return_value = tmp_path / "ds"
        bad_result = ValidationResult()
        bad_result.add_error("missing data/")
        mock_val.return_value = bad_result

        with pytest.raises(DatasetValidationError) as exc_info:
            prepare_datasets(["TestUser/ds"], cache_root=tmp_path)
        assert "missing data/" in str(exc_info.value)

    @patch("hf.dataset_hub.download_dataset")
    @patch("hf.dataset_hub.validate_dataset")
    def test_multi_dataset_merges(self, mock_val, mock_dl, tmp_path):
        ds1 = tmp_path / "ds1"
        ds2 = tmp_path / "ds2"
        merge_out = tmp_path / "merged"
        mock_dl.side_effect = [ds1, ds2]
        mock_val.return_value = ValidationResult(valid=True)

        with patch.dict("sys.modules", {"merge_datasets": MagicMock()}) as _:
            import importlib
            merge_mod = sys.modules["merge_datasets"]
            merge_mod.merge = MagicMock()

            result = prepare_datasets(
                ["TestUser/ds1", "TestUser/ds2"],
                cache_root=tmp_path,
                merge_output=merge_out,
            )

            merge_mod.merge.assert_called_once()
            call_kwargs = merge_mod.merge.call_args[1]
            assert call_kwargs["source_roots"] == [ds1, ds2]
            assert call_kwargs["output_root"] == merge_out

        assert mock_dl.call_count == 2
