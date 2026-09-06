import csv
from pathlib import Path

OLD_MANIFEST_PATH = (
    r"D:\Documenti\Università\Computer Science\Deep Learning"
    r"\project\manifests\all_tracks_800.csv"
)

NEW_MANIFEST_PATH = (
    r"D:\Documenti\Università\Computer Science\Deep Learning"
    r"\project\manifests\all_tracks.csv"
)

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

old_rows = load_rows(OLD_MANIFEST_PATH)
new_rows = load_rows(NEW_MANIFEST_PATH)

old_slakh_splits = {
    row["path"]: row["split"]
    for row in old_rows
    if row["source"] == "slakh"
    and row["split"] in {"train", "val"}
}

new_slakh_splits = {
    row["path"]: row["split"]
    for row in new_rows
    if row["source"] == "slakh"
    and row["split"] in {"train", "val"}
}

missing_old_paths = set(old_slakh_splits) - set(new_slakh_splits)

changed_old_splits = {
    path
    for path in old_slakh_splits
    if path in new_slakh_splits
    and old_slakh_splits[path] != new_slakh_splits[path]
}

new_800_to_1069_val = sum(
    row["source"] == "slakh"
    and row["split"] == "val"
    and 800 <= int(Path(row["path"]).stem.replace("train_", "")) <= 1069
    for row in new_rows
)

new_1070_to_1394_val = sum(
    row["source"] == "slakh"
    and row["split"] == "val"
    and 1070 <= int(Path(row["path"]).stem.replace("train_", "")) <= 1394
    for row in new_rows
)

print("Missing original Slakh rows:", len(missing_old_paths))
print("Changed original Slakh splits:", len(changed_old_splits))
print("Validation tracks in 800–1069:", new_800_to_1069_val)
print("Validation tracks in 1070–1394:", new_1070_to_1394_val)

assert not missing_old_paths
assert not changed_old_splits
assert new_800_to_1069_val == 35
assert new_1070_to_1394_val == 35

print("PASS: original 800-track Slakh splits are preserved.")
print("PASS: both newly added Slakh blocks have 35 validation tracks.")