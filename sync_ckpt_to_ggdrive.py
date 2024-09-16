import os
import json
import time
import argparse
from ggdrive_cli.ggdrive import upload_single
from ggdrive_cli.file_utils import list_files


def wait_until_file_statble(file_path, interval=0.5):
    """File is considered stable if its size does not change after `interval` seconds."""

    try:
        previous_size = os.path.getsize(file_path)
        while True:
            time.sleep(interval)
            current_size = os.path.getsize(file_path)
            if current_size == previous_size:
                return True
            previous_size = current_size
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch_dir", "-f", required=True)
    parser.add_argument("--sync_tracker_file", default="sync_tracker.json")
    parser.add_argument("--parent_id", default=None)
    args = parser.parse_args()

    if os.path.exists(args.sync_tracker_file):
        with open(args.sync_tracker_file, "r", encoding="utf-8") as reader:
            already_synced = json.load(reader)
    else:
        already_synced = {}

    iterating_status_tracker = {}
    while True:
        paths = list_files(args.watch_dir)[1:]
        for path in paths:
            if path not in already_synced:
                if os.path.isfile(path):
                    done = wait_until_file_statble(path)
                    if not done:
                        continue
                already_synced[path] = {"status": True}
                path_dir = os.path.dirname(path)
                parent_id = already_synced.get(path_dir, {}).get(
                    "folder_id", args.parent_id
                )
                id_tracker = {}
                upload_single(
                    path,
                    parent_id=parent_id,
                    id_tracker=id_tracker,
                    iterating_status_tracker=iterating_status_tracker,
                )
                if id_tracker:
                    folder_id = id_tracker[path]
                    already_synced[path]["folder_id"] = folder_id
                with open(args.sync_tracker_file, "w", encoding="utf-8") as writer:
                    json.dump(already_synced, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
