import os
import sys
import urllib.request

os.makedirs("data", exist_ok=True)

SIZE = sys.argv[1] if len(sys.argv) > 1 else "medium"
SIZES = {
    "small" : 2_000_000,
    "medium": 5_000_000,
    "large" : 10_000_000,
    "full"  : 50_000_000,
    "max"   : 100_000_000,
}

if SIZE not in SIZES:
    print("Unknown size:", SIZE)
    print("Valid options:", list(SIZES.keys()))
    sys.exit(1)

LIMIT = SIZES[SIZE]
URL = (
    "https://huggingface.co/datasets/roneneldan/"
    "TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
)
SAVE = "data/tinystories.txt"

mb = LIMIT // 1_000_000
print("=" * 45)
print("  TinyStories Downloader")
print("=" * 45)
print("  Size   :", SIZE, "=", mb, "MB")
print("  Save to:", SAVE)
print("=" * 45)
print()
print("Downloading... please wait...")
print()


def show_progress(count, block_size, total_size):
    downloaded = count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        done = int(pct / 2)
        bar = "#" * done + "-" * (50 - done)
        dl_mb = downloaded / 1_000_000
        print(
            "\r  [" + bar + "] "
            + str(round(pct, 1)) + "% "
            + str(round(dl_mb, 1)) + "MB",
            end="",
            flush=True
        )
    else:
        dl_mb = downloaded / 1_000_000
        print(
            "\r  Downloaded: " + str(round(dl_mb, 1)) + "MB",
            end="",
            flush=True
        )


try:
    tmp = SAVE + ".tmp"
    urllib.request.urlretrieve(URL, tmp, reporthook=show_progress)
    print()
    print()

    print("Trimming to", mb, "MB...")
    with open(tmp, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read(LIMIT)

    with open(SAVE, "w", encoding="utf-8") as f:
        f.write(content)

    os.remove(tmp)

    chars  = len(content)
    words  = len(content.split())
    tokens = int(words * 1.3)
    size_mb = round(chars / 1_000_000, 1)

    print()
    print("=" * 45)
    print("  Download Complete!")
    print("=" * 45)
    print("  File  :", SAVE)
    print("  Size  :", size_mb, "MB")
    print("  Chars :", chars)
    print("  Words :", words)
    print("  Tokens:", tokens)
    print("=" * 45)

    if tokens < 1_000_000:
        print("  Risk  : HIGH - need more data")
    elif tokens < 5_000_000:
        print("  Risk  : MEDIUM - acceptable")
    elif tokens < 15_000_000:
        print("  Risk  : LOW - good!")
    else:
        print("  Risk  : MINIMAL - excellent!")

    print()
    print("Now run: python main.py")
    print()

except KeyboardInterrupt:
    print()
    print("Cancelled!")
    if os.path.exists(SAVE + ".tmp"):
        os.remove(SAVE + ".tmp")
    sys.exit(0)

except Exception as e:
    print()
    print("Download failed:", e)
    print("Creating fallback dataset...")

    story = (
        "Once upon a time there was a little girl "
        "named Lily. She loved to play in the garden "
        "every day with her friends. "
        "One day she found a small puppy near a big tree. "
        "The puppy had big brown eyes and a wagging tail. "
        "Lily took the puppy home and they became best friends.\n\n"
        "<|endoftext|>\n\n"
        "There was a little boy named Tim who loved "
        "to explore the big forest near his house. "
        "One sunny morning he found a baby bird "
        "that had fallen from its nest. "
        "Tim carefully put it back and the bird sang for him.\n\n"
        "<|endoftext|>\n\n"
        "In a small village there lived a kind old wizard. "
        "Every day children came to hear his magical stories. "
        "One day a tiny dragon knocked on his door "
        "and asked to learn magic too. "
        "The wizard smiled and said yes.\n\n"
        "<|endoftext|>\n\n"
        "A brave girl named Sara lived near a big forest. "
        "She was never afraid of anything at all. "
        "One day she found a small fox stuck in a branch. "
        "She helped it get free and the fox became her friend.\n\n"
        "<|endoftext|>\n\n"
    )

    repeats = max(1, LIMIT // len(story) + 1)
    content = (story * repeats)[:LIMIT]

    with open(SAVE, "w", encoding="utf-8") as f:
        f.write(content)

    print("Fallback created:", round(len(content) / 1_000_000, 1), "MB")
    print()
    print("Note: Fallback is repetitive.")
    print("Try downloading again later for better results.")
    print()
    print("Now run: python main.py")