"""
Downloads the UCI Student Performance dataset and prepares it for
train_models.py: derives a binary `pass` target from the final grade
(G3 >= 10 out of 20) and drops G1/G2/G3 so the model predicts from the
30 behavioral/demographic/study-habit features only, not from earlier
grades (this is the "hardest" of the three prediction settings the
dataset's own documentation describes, and avoids the G3 correlation
acting as near-leakage).

Source: https://archive.ics.uci.edu/dataset/320/student+performance
(uses the Portuguese-course file, student-por.csv: 649 students, 30
features after dropping G1/G2/G3 — well above the assignment's minimum
of 12 features / 500 instances).

Usage: python prepare_student_data.py
Produces: data/student_performance.csv
"""

import io
import os
import urllib.request
import zipfile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
OUT_PATH = os.path.join("data", "student_performance.csv")
PASS_THRESHOLD = 10  # G3 (final grade, 0-20) >= 10 counts as a pass


def download_student_por():
    with urllib.request.urlopen(URL) as response:
        outer_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer_zip:
        inner_bytes = outer_zip.read("student.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
        with inner_zip.open("student-por.csv") as f:
            return pd.read_csv(f, sep=";")


def main():
    os.makedirs("data", exist_ok=True)
    df = download_student_por()

    df["pass"] = (df["G3"] >= PASS_THRESHOLD).astype(int)
    df = df.drop(columns=["G1", "G2", "G3"])

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1] - 1} features -> {OUT_PATH}")
    print(f"Pass rate: {df['pass'].mean():.1%}")


if __name__ == "__main__":
    main()
