# Read lines.txt and save cleaned transcriptions to train.txt
with open("lines.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Store cleaned transcriptions
cleaned_lines = []

for line in lines:
    line = line.strip()
    if not line or line.startswith("#"):
        continue  # Skip empty lines and comments

    parts = line.split()
    if len(parts) < 9:
        continue  # Skip malformed lines

    # Extract transcription and replace | with space (use all lines)
    transcription = " ".join(parts[8:]).replace("|", " ")
    cleaned_lines.append(transcription)

# Save all cleaned lines into a single file
with open("train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

print(f"Saved {len(cleaned_lines)} training lines to train.txt")
