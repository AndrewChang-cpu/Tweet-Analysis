import os
import glob
import matplotlib.pyplot as plt

# Directory where the part files are stored
data_dir = os.path.expanduser("~/output_tweets_features.csv")  # Change if needed

# Find all part-*.csv files
csv_files = sorted(glob.glob(os.path.join(data_dir, "part-*.csv")))

# Extract favorite counts (3rd from last column in each line)
favorite_counts = []

total_lines = 0  # Counter for total lines processed
skipped_lines = 0

for path in csv_files:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split from the right to handle commas in the text column
            parts = line.rsplit(",", 6)
            if len(parts) != 7:
                if skipped_lines < 25:
                    print(parts)
                skipped_lines += 1
                continue  # Skip malformed rows
            try:
                fav_count = int(parts[-5])  # 3rd from last = index -5
                if fav_count > 0:  # Exclude entries with favorite count of 0
                    favorite_counts.append(fav_count)
                total_lines += 1
            except ValueError:
                continue  # Skip non-integer entries

print(f"Total lines processed: {total_lines}")
print(f"Skipped lines: {skipped_lines}")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(favorite_counts, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Favorite Counts")
plt.xlabel("Favorite Count")
plt.ylabel("Frequency")
plt.grid(True)

# Save image
plt.savefig("favorite_count_histogram.png")
plt.close()

print("Saved histogram as favorite_count_histogram.png")

