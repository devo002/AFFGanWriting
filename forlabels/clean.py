import re

# Choose your input and output files
input_file = "train.txt"        # or "merged.txt"
output_file = "train_cleaned.txt"

# Read and clean
with open(input_file, "r", encoding="utf-8") as fin:
    lines = fin.readlines()

cleaned_lines = []
for line in lines:
    # Remove space before punctuation (keep space after)
    cleaned = re.sub(r"\s+([.,!?;:\"”’)])", r"\1", line)
    cleaned_lines.append(cleaned.strip())

# Write cleaned output
with open(output_file, "w", encoding="utf-8") as fout:
    fout.write("\n".join(cleaned_lines))

print(f"Cleaned {len(cleaned_lines)} lines and saved to {output_file}")
