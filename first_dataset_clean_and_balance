import re
import pandas as pd

INPUT_FILE = "/Users/hasancan/Downloads/messy_irony_data.csv"
OUTPUT_FILE = "balanced_dataset_adjusted.csv"

TEXT_COL = "text"
LABEL_COL = "label"
LITERAL_LABEL = 0
IRONIC_LABEL = 1

# literal constraints
MIN_WORDS_LITERAL = 5
MAX_WORDS_LITERAL = 50
REQUIRE_SINGLE_SENTENCE_LITERAL = True
REMOVE_NEWLINES_LITERAL = True

# ironic cleanup
NORMALIZE_NEWLINES_IRONIC = True  # turn \n into space instead of dropping

# how much difference allowed between classes (0.10 = 10%)
ALLOWED_DIFF_RATIO = 0.10

def count_sentences(text: str) -> int:
    parts = re.split(r'[.!?]+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)

# 1) load
df = pd.read_csv(INPUT_FILE)

# 2) split
literal_df = df[df[LABEL_COL] == LITERAL_LABEL].copy()
ironic_df = df[df[LABEL_COL] == IRONIC_LABEL].copy()

# 3) CLEAN LITERALS
if REMOVE_NEWLINES_LITERAL:
    literal_df = literal_df[~literal_df[TEXT_COL].str.contains(r'[\n\r]', regex=True)]

# add counts
literal_df["word_count"] = literal_df[TEXT_COL].str.split().str.len()
literal_df["sent_count"] = literal_df[TEXT_COL].apply(count_sentences)

# apply filters
literal_mask = (literal_df["word_count"] >= MIN_WORDS_LITERAL) & \
               (literal_df["word_count"] <= MAX_WORDS_LITERAL)

if REQUIRE_SINGLE_SENTENCE_LITERAL:
    literal_mask &= (literal_df["sent_count"] == 1)

literal_df = literal_df[literal_mask].copy()

# drop helper cols
literal_df = literal_df.drop(columns=["word_count", "sent_count"])

print(f"[INFO] Literals after cleaning: {len(literal_df)}")

# 4) CLEAN IRONICS
if NORMALIZE_NEWLINES_IRONIC:
    ironic_df[TEXT_COL] = ironic_df[TEXT_COL].str.replace(r'[\n\r]+', ' ', regex=True)
else:
    ironic_df = ironic_df[~ironic_df[TEXT_COL].str.contains(r'[\n\r]', regex=True)]

print(f"[INFO] Ironics after cleaning: {len(ironic_df)}")

# 5) BALANCE WITH TOLERANCE
literal_count = len(literal_df)
ironic_count = len(ironic_df)

# target is to get them as close as possible, but allow 5-10% diff
# we'll downsample the bigger one
bigger_label = None

if literal_count == 0 or ironic_count == 0:
    raise ValueError("One of the classes is empty after filtering. Loosen your filters.")

# compute allowed diff
max_allowed_diff = int(min(literal_count, ironic_count) * ALLOWED_DIFF_RATIO)

if abs(literal_count - ironic_count) <= max_allowed_diff:
    # close enough, do nothing
    print(f"[INFO] Counts already within {ALLOWED_DIFF_RATIO*100:.0f}% tolerance.")
else:
    # need to downsample the bigger group
    if literal_count > ironic_count:
        bigger_label = "literal"
        # downsample literals
        literal_df = literal_df.sample(n=ironic_count, random_state=42)
    else:
        bigger_label = "ironic"
        ironic_df = ironic_df.sample(n=literal_count, random_state=42)
    print(f"[INFO] Downsampled {bigger_label} to match the smaller class.")

# 6) combine
final_df = pd.concat([literal_df, ironic_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 7) show end result
final_literal = len(final_df[final_df[LABEL_COL] == LITERAL_LABEL])
final_ironic = len(final_df[final_df[LABEL_COL] == IRONIC_LABEL])

print("========== FINAL COUNTS ==========")
print(f"Literals (label {LITERAL_LABEL}): {final_literal}")
print(f"Ironics  (label {IRONIC_LABEL}): {final_ironic}")
diff = abs(final_literal - final_ironic)
print(f"Difference: {diff} ({(diff / min(final_literal, final_ironic)) * 100:.2f}% of smaller class)")
print("===================================")

# peek
print("\nSample rows:")
print(final_df.head(10))

# 8) save
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"[INFO] Saved to {OUTPUT_FILE}")
