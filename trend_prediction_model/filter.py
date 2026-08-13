"""
CMS Part D — Prescribers by Provider and Drug
Reusable preprocessing pipeline

WHAT THIS SCRIPT DOES
----------------------
1. Reads the huge raw CSV in CHUNKS (never loads all 4GB into memory at once)
2. Keeps only the columns we actually need
3. Aggregates from prescriber-level rows -> drug-level rows
   (millions of rows -> a few thousand rows)
4. Handles suppressed/missing values (CMS blanks out small counts for privacy)
5. Filters down to the top N drugs by total claims (keeps the dataset focused)
6. Adds a `year` column so multiple years can later be stacked together
7. Saves a small, clean CSV per year into processed_data/

HOW TO USE FOR MULTIPLE YEARS
------------------------------
Just call process_year() once per year, pointing at that year's raw file.
Example is at the bottom of this script (the "if __name__" block).

WHY CHUNKING MATTERS
---------------------
The raw file is ~4GB. Trying to do pd.read_csv("huge_file.csv") directly
will either crash your machine or take forever. Reading in chunks means
we only ever hold e.g. 500,000 rows in memory at a time, filter/aggregate
that chunk, throw it away, and move to the next chunk.
"""

import pandas as pd
import os

# ----------------------------------------------------------------------
# CONFIG — adjust these if needed
# ----------------------------------------------------------------------

# Only load these columns from the raw file (saves memory — we don't need
# prescriber name, city, NPI etc. for a drug-level trend model)
COLUMNS_TO_KEEP = [
    "Gnrc_Name",       # Generic drug name (better for matching than brand name)
    "Brnd_Name",       # Brand name (kept for reference/display)
    "Tot_Clms",        # Total claims (refills included) -> MAIN target variable
    "Tot_Drug_Cst",    # Total cost paid -> SECOND target variable
    "Tot_Benes",       # Total unique beneficiaries (patients)
    "Tot_30day_Fills", # Standardized 30-day fills
    "Tot_Day_Suply",   # Total days supplied
]

CHUNK_SIZE = 500_000       # rows per chunk — safe for most laptops
TOP_N_DRUGS = None         # set to an integer (e.g. 150) to keep only the
                            # top N by claims, or None to keep ALL drugs
                            # that survive CMS's own suppression rule
OUTPUT_DIR = "processed_data"


def process_year(raw_filepath: str, year: int, fixed_drug_list=None) -> pd.DataFrame:
    """
    Process one year's raw CMS Part D file into a clean, drug-level dataset.

    Parameters
    ----------
    raw_filepath : path to the raw CSV file for this year (the big ~4GB file)
    year : the year this file corresponds to (e.g. 2024)
    fixed_drug_list : optional list of Gnrc_Name values. If provided, the
        output ALWAYS contains exactly this set of drugs (not the top N
        for this particular year). This is essential for multi-year trend
        data — see note below.

        WHY THIS MATTERS: if you pick "top 150 by claims" independently
        for each year, the actual 150 drugs can differ year to year (a
        drug might rank #148 in 2021 but drop out of the top 150 by 2024).
        That means many drugs end up with data for only 1-2 years instead
        of all years, which breaks lag-feature creation for trend
        prediction. Passing a fixed_drug_list (usually taken from your
        most recent/reference year) guarantees every year has data for
        the exact same drugs, even if a drug's rank moved.

    Returns
    -------
    A small pandas DataFrame, aggregated to one row per drug, with a
    'year' column added. Also saves this DataFrame as a CSV.
    """

    print(f"\n=== Processing year {year} ===")
    print(f"Reading from: {raw_filepath}")

    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(
            f"Could not find {raw_filepath}. "
            f"Make sure the file is downloaded and the path is correct."
        )

    # This will hold the running total for each drug as we process chunk by chunk
    drug_totals = {}

    chunk_num = 0
    total_rows_seen = 0

    # ----------------------------------------------------------------
    # STEP 1: Read the file in chunks, aggregate as we go
    # ----------------------------------------------------------------
    for chunk in pd.read_csv(
        raw_filepath,
        usecols=COLUMNS_TO_KEEP,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="latin1",
    ):
        chunk_num += 1
        total_rows_seen += len(chunk)

        # --- Handle suppressed/missing values ---
        # CMS blanks out Tot_Benes when the count is < 11 (privacy rule).
        # We can't know the exact number, so we impute a small placeholder (5)
        # rather than dropping the row entirely (dropping would bias us
        # toward only high-volume drugs before we've even aggregated).
        chunk["Tot_Benes"] = chunk["Tot_Benes"].fillna(5)

        # Some numeric columns might read in as strings if there are
        # stray blanks — force them to numeric, turning bad values into NaN,
        # then fill with 0 (0 claims/cost contributes nothing to the sum).
        numeric_cols = ["Tot_Clms", "Tot_Drug_Cst", "Tot_Benes",
                         "Tot_30day_Fills", "Tot_Day_Suply"]
        for col in numeric_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

        # Drop rows with no generic name at all (can't aggregate what we can't identify)
        chunk = chunk.dropna(subset=["Gnrc_Name"])

        # --- Aggregate this chunk by generic drug name ---
        chunk_agg = chunk.groupby("Gnrc_Name").agg(
            Brnd_Name=("Brnd_Name", "first"),   # just keep one brand name for reference
            Tot_Clms=("Tot_Clms", "sum"),
            Tot_Drug_Cst=("Tot_Drug_Cst", "sum"),
            Tot_Benes=("Tot_Benes", "sum"),
            Tot_30day_Fills=("Tot_30day_Fills", "sum"),
            Tot_Day_Suply=("Tot_Day_Suply", "sum"),
        )

        # --- Merge this chunk's totals into our running grand total ---
        for gnrc_name, row in chunk_agg.iterrows():
            if gnrc_name not in drug_totals:
                drug_totals[gnrc_name] = {
                    "Brnd_Name": row["Brnd_Name"],
                    "Tot_Clms": 0,
                    "Tot_Drug_Cst": 0.0,
                    "Tot_Benes": 0,
                    "Tot_30day_Fills": 0,
                    "Tot_Day_Suply": 0,
                }
            drug_totals[gnrc_name]["Tot_Clms"] += row["Tot_Clms"]
            drug_totals[gnrc_name]["Tot_Drug_Cst"] += row["Tot_Drug_Cst"]
            drug_totals[gnrc_name]["Tot_Benes"] += row["Tot_Benes"]
            drug_totals[gnrc_name]["Tot_30day_Fills"] += row["Tot_30day_Fills"]
            drug_totals[gnrc_name]["Tot_Day_Suply"] += row["Tot_Day_Suply"]

        print(f"  Chunk {chunk_num} processed — {total_rows_seen:,} rows seen so far, "
              f"{len(drug_totals):,} unique drugs so far")

    # ----------------------------------------------------------------
    # STEP 2: Convert the running totals dict into a clean DataFrame
    # ----------------------------------------------------------------
    result = pd.DataFrame.from_dict(drug_totals, orient="index")
    result.index.name = "Gnrc_Name"
    result = result.reset_index()

    # ----------------------------------------------------------------
    # STEP 3: Filter down to the relevant drugs
    # ----------------------------------------------------------------
    if fixed_drug_list is not None:
        # Use the SAME set of drugs across every year (see docstring above)
        result = result[result["Gnrc_Name"].isin(fixed_drug_list)]
        missing = set(fixed_drug_list) - set(result["Gnrc_Name"])
        if missing:
            print(f"  WARNING: {len(missing)} drug(s) from the fixed list "
                  f"had no claims in {year} (below suppression threshold "
                  f"or genuinely not prescribed that year): {sorted(missing)[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
        result = result.sort_values("Tot_Clms", ascending=False)
    else:
        # No fixed list given
        if TOP_N_DRUGS is None:
            # Keep EVERY drug that survived CMS's own suppression rule
            # (claims >= 11) — no artificial cutoff at all.
            result = result.sort_values("Tot_Clms", ascending=False)
        else:
            # Take only the top N by claims for this year
            result = result.sort_values("Tot_Clms", ascending=False).head(TOP_N_DRUGS)

    # ----------------------------------------------------------------
    # STEP 4: Add the year column — this is what makes multi-year
    # stacking possible later
    # ----------------------------------------------------------------
    result["year"] = year

    # ----------------------------------------------------------------
    # STEP 5: Save to a small CSV
    # ----------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"cms_partd_{year}_clean.csv")
    result.to_csv(output_path, index=False)

    print(f"Done. Saved {len(result)} drugs to {output_path}")
    print(result.head(10).to_string(index=False))

    return result


def combine_years(output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Once you've run process_year() for multiple years, call this to
    stack all the yearly clean CSVs into one combined dataset —
    this combined table is what you'll actually train the model on.
    """
    # IMPORTANT: only match files that look like "cms_partd_<YEAR>_clean.csv"
    # (e.g. cms_partd_2024_clean.csv). This deliberately excludes files like
    # "cms_partd_combined.csv" or "cms_partd_combined_clean.csv" — outputs
    # from a PREVIOUS run of this same function — which would otherwise get
    # re-ingested as if they were a new year's data and silently double-count
    # everything. Always re-run combine_years() on a folder that only
    # contains the individual yearly files, or rely on this stricter filter.
    all_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("cms_partd_")
        and f.endswith("_clean.csv")
        and f not in ("cms_partd_combined.csv", "cms_partd_combined_clean.csv")
        and f[len("cms_partd_"):-len("_clean.csv")].isdigit()  # the middle part must be a year, e.g. "2024"
    ]

    if not all_files:
        raise FileNotFoundError(
            f"No processed yearly files found in {output_dir}. "
            f"Run process_year() for at least one year first."
        )

    dfs = [pd.read_csv(f) for f in all_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["Gnrc_Name", "year"])

    combined_path = os.path.join(output_dir, "cms_partd_combined.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\nCombined {len(all_files)} year(s) into {combined_path}")
    print(f"Total rows: {len(combined)} | Years included: {sorted(combined['year'].unique())}")

    return combined


# ----------------------------------------------------------------------
# HOW TO RUN THIS — FULL SCALE (ALL ~1,800 DRUGS, NOT JUST TOP 150)
# ----------------------------------------------------------------------
# IMPORTANT: TOP_N_DRUGS is now set to None above, so every drug that
# survives CMS's own suppression rule will be kept for EVERY year. This
# means the fixed_drug_list trick is less critical than before (we're not
# picking an arbitrary top-N per year anymore), but we still recommend
# using it OR just relying on combine_years() to filter down to drugs
# present in ALL years afterward (same approach as before).
#
# EXPECTATION: raw drug counts per year were around 1,683-1,823 (you
# already measured this). After filtering to only drugs present in EVERY
# year (2019-2024), expect meaningfully fewer — many low-volume drugs
# will be missing in at least one year, since CMS excludes any
# drug-year combination with fewer than 11 total claims from the file
# entirely (not just blanked out — the row doesn't exist at all that
# year). A realistic estimate is somewhere in the 600-1,000 range, but
# the exact number will only be known once you run it.
if __name__ == "__main__":

    # --- Process every year — update these filenames to your actual files ---
    process_year(raw_filepath="dataset/2024/data_2024.csv", year=2024)
    process_year(raw_filepath="dataset/2023/data_2023.csv", year=2023)
    process_year(raw_filepath="dataset/2022/data_2022.csv", year=2022)
    process_year(raw_filepath="dataset/2021/data_2021.csv", year=2021)
    process_year(raw_filepath="dataset/2020/data_2020.csv", year=2020)
    process_year(raw_filepath="dataset/2019/data_2019.csv", year=2019)

    # --- Combine all processed years into one file ---
    combine_years()

    # --- Filter down to only drugs present in EVERY year (same approach
    # as before — this is what makes lag features possible for every row) ---
    import pandas as pd
    combined = pd.read_csv(os.path.join(OUTPUT_DIR, "cms_partd_combined.csv"))
    years_count = combined["year"].nunique()
    counts = combined.groupby("Gnrc_Name")["year"].count()
    complete_drugs = counts[counts == years_count].index
    final = combined[combined["Gnrc_Name"].isin(complete_drugs)].sort_values(["Gnrc_Name", "year"])
    final_path = os.path.join(OUTPUT_DIR, "cms_partd_combined_clean.csv")
    final.to_csv(final_path, index=False)

    print(f"\n=== FINAL RESULT ===")
    print(f"Drugs with complete {years_count}-year history: {len(complete_drugs)}")
    print(f"Total rows in final file: {len(final)}")
    print(f"Saved to: {final_path}")