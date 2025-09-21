# Load sets of base line IDs
def load_base_ids(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

train_ids = load_base_ids("linestrain.txt")
val_ids = load_base_ids("linesvalidation.txt")
test_ids = load_base_ids("linestest.txt")

# Prepare output lists
train_lines = []
val_lines = []
test_lines = []
unmatched_lines = []

# Process merged_cleaned.txt
with open("merged_cleaned.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        full_id = line.split("|")[0]
        base_id = "-".join(full_id.split("-")[:2])  # e.g., a01-000u

        if base_id in train_ids:
            train_lines.append(line)
        elif base_id in val_ids:
            val_lines.append(line)
        elif base_id in test_ids:
            test_lines.append(line)
        else:
            unmatched_lines.append(line)

# Save to files
with open("train_merged.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))

with open("val_merged.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val_lines))

with open("test_merged.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(test_lines))

with open("unmatched_merged.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(unmatched_lines))

# Summary
print(f"✅ Saved:")
print(f"  - {len(train_lines)} lines to train_merged.txt")
print(f"  - {len(val_lines)} lines to val_merged.txt")
print(f"  - {len(test_lines)} lines to test_merged.txt")
print(f"  - {len(unmatched_lines)} lines to unmatched_merged.txt (not in any list)")
