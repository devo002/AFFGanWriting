import re

# Read both files
with open("full_line_ids.txt", "r", encoding="utf-8") as f:
    line_ids = [line.strip() for line in f if line.strip()]

with open("train.txt", "r", encoding="utf-8") as f:
    transcriptions = [line.strip() for line in f if line.strip()]

# Check matching lengths
assert len(line_ids) == len(transcriptions), "Mismatch between line IDs and transcriptions!"

# Clean and merge
merged_lines = []
for line_id, text in zip(line_ids, transcriptions):
    # Remove space before punctuation (e.g., "word ." → "word.")
    cleaned_text = re.sub(r"\s+([.,!?;:\"”’)])", r"\1", text)
    merged_lines.append(f"{line_id}|{cleaned_text}")

# Save to merged file
with open("merged_cleaned.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(merged_lines))

print(f"Merged and cleaned {len(merged_lines)} lines to merged_cleaned.txt")
