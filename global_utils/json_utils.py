import json
import orjson

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def _load_one(path: Path) -> tuple[str, dict]:
    """
    Worker: read the file's bytes and parse with orjson.
    Returns (filename_without_extension, parsed_dict).
    """
    try:
        raw = path.read_bytes()
        return path.stem, orjson.loads(raw)
    except Exception as e:
        print(f"[warn] Unable to load {path.name}: {e}")
        return path.stem, {}

def load_all_jsons(folder: str | Path, max_workers: int | None = None) -> dict[str, dict]:
    """
    Load every .json in `folder` in parallel using orjson + processes.

    Args:
        folder: directory containing .json files
        max_workers: number of worker processes (defaults to os.cpu_count())

    Returns:
        Dict mapping filename (no ".json") → parsed JSON as dict
    """
    folder = Path(folder)
    json_files = list(folder.glob("*.json"))
    results: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit all files to the pool
        futures = {executor.submit(_load_one, path): path for path in json_files}
        # as they complete, collect results
        for fut in as_completed(futures):
            name, data = fut.result()
            results[name] = data
    
    results = {k:v for k,v in results.items() if v}
    return results

def _write_one(folder: Path, name: str, data: dict) -> Path:
    """
    Worker: serialize with orjson and write to disk.
    Returns the path written.
    """
    path = folder / f"{name}.json"
    raw = orjson.dumps(data)
    path.write_bytes(raw)
    return path

def write_all_jsons(
    data_map: dict[str, dict],
    folder: str | Path,
    max_workers: int | None = None
) -> list[Path]:
    """
    Write each JSON in `data_map` to `folder` in parallel using orjson.

    Args:
        data_map: mapping filename (no ".json") → JSON-serializable dict
        folder: target directory for .json files
        max_workers: number of worker processes (defaults to os.cpu_count())

    Returns:
        List of Paths for the files written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_write_one, folder, name, data): name
            for name, data in data_map.items()
        }
        for fut in as_completed(futures):
            # will re-raise exceptions if any write failed
            results.append(fut.result())

    return results