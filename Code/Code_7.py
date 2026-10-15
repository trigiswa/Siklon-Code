# ===== CELL 0 =====
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, norm
import gc

print("Starting...")

# ============================================================
# 1. FILE LOCATION
# ============================================================

FILE = Path("Siklon_Tropis_ERA5_Extracted.xlsx")

if not FILE.exists():
    raise FileNotFoundError(
        f"File not found:\n{FILE.resolve()}\n\n"
        "Make sure the Excel file is located in the Jupyter working folder."
    )

print("File found:")
print(FILE.resolve())


# ============================================================
# 2. READ EXCEL FILE
# ============================================================

df = pd.read_excel(FILE)

print("\nData berresults dibaca.")
print("Number of rows :", len(df))
print("Number of columns :", len(df.columns))


# ============================================================
# 3. CHECK COLUMNS
# ============================================================

print("\nAvailable columns:")

for col in df.columns:
    print(" -", col)


# ============================================================
# 4. CHECK COLUMNS WAJIB
# ============================================================

required = [
    "SID",
    "ISO_TIME",
    "LAT",
    "LON",
    "era5_sst",
    "era5_tp",
    "era5_w",
    "era5_vo",
    "WIND_km/h",
    "PRESSURE_hPa"
]

missing_columns = [
    col for col in required
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nThe following columns were not found:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required columns are available.")


# ============================================================
# 5. TIME CONVERSION
# ============================================================

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

jumlah_waktu_invalid = df["ISO_TIME"].isna().sum()

print(
    "\nUnreadable time values :",
    jumlah_waktu_invalid
)


# ============================================================
# 6. DROP ROWS WITHOUT SID / TIME
# ============================================================

df = df.dropna(
    subset=["SID", "ISO_TIME"]
).copy()

print(
    "Rows after cleaning :",
    len(df)
)


# ============================================================
# 7. SORT BY SID AND TIME
# ============================================================

df = df.sort_values(
    ["SID", "ISO_TIME"]
)


# ============================================================
# 8. TAKE ONE GENESIS POINT PER SID
# ============================================================

# Earliest point in time for each SID

idx_genesis = (
    df.groupby("SID")["ISO_TIME"]
      .idxmin()
)

genesis = (
    df.loc[idx_genesis]
      .copy()
      .sort_values("ISO_TIME")
      .reset_index(drop=True)
)


# ============================================================
# 9. INITIAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GENESIS IDENTIFICATION RESULTS")
print("=" * 60)

print(
    "Number of unique SIDs in the raw values :",
    df["SID"].nunique()
)

print(
    "Number of genesis points :",
    len(genesis)
)


# ============================================================
# 10. FREE MEMORY
# ============================================================

del idx_genesis
gc.collect()

print("\nCELL 1 SELESAI.")

# ===== CELL 1 =====
# ============================================================
# CELL 2
# STUDY DOMAIN SELECTION
# ============================================================

print("Starting domain selection...")


# ============================================================
# 1. COPY GENESIS DATA
# ============================================================

genesis_domain = genesis.copy()


# ============================================================
# 2. STUDY DOMAIN
# ============================================================

genesis_domain = genesis_domain[
    (genesis_domain["LON"] >= 95) &
    (genesis_domain["LON"] <= 145) &
    (genesis_domain["LAT"] >= -15) &
    (genesis_domain["LAT"] <= 25)
].copy()


# ============================================================
# 3. RESET INDEX
# ============================================================

genesis_domain = (
    genesis_domain
    .reset_index(drop=True)
)


# ============================================================
# 4. ZONE CLASSIFICATION
# ============================================================

def classify_zone(lat):

    if -5 <= lat <= 5:
        return "Equatorial"

    elif lat > 5:
        return "North"

    else:
        return "South"


genesis_domain["ZONE"] = (
    genesis_domain["LAT"]
    .apply(classify_zone)
)


# ============================================================
# 5. HASIL
# ============================================================

print("\n" + "=" * 60)
print("DOMAIN SELECTION RESULTS")
print("=" * 60)

print(
    "Number of genesis cases in the domain :",
    len(genesis_domain)
)


print("\nZone distribution:")

print(
    genesis_domain["ZONE"]
    .value_counts()
)


# ============================================================
# 6. CEK VAMEI DAN SENYAR
# ============================================================

target = genesis_domain[
    genesis_domain["NAME"]
    .astype(str)
    .str.upper()
    .isin(["VAMEI", "SENYAR"])
].copy()


print("\n" + "=" * 60)
print("VAMEI DAN SENYAR")
print("=" * 60)

print(
    target[
        [
            "SID",
            "NAME",
            "SEASON",
            "LAT",
            "LON",
            "ZONE"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. VALIDASI JUMLAH
# ============================================================

if len(genesis_domain) == 392:

    print(
        "\n✓ CORRECT: the research valuesset contains 392 cyclones."
    )

else:

    print(
        "\n⚠ WARNING!"
    )

    print(
        "Obtained total:",
        len(genesis_domain)
    )

    print(
        "Kita JANGAN lanjut ke model sebelum "
        "penyebab perbedaan ini diperiksa."
    )


# ============================================================
# 8. VALIDASI KOMPOSISI ZONA
# ============================================================

jumlah_utara = (
    genesis_domain["ZONE"] == "North"
).sum()

jumlah_ekuator = (
    genesis_domain["ZONE"] == "Equatorial"
).sum()

jumlah_selatan = (
    genesis_domain["ZONE"] == "South"
).sum()


print("\nComposition:")
print("North      :", jumlah_utara)
print("Equatorial :", jumlah_ekuator)
print("South    :", jumlah_selatan)

print(
    "\nTotal       :",
    jumlah_utara
    + jumlah_ekuator
    + jumlah_selatan
)


print("\nCELL 2 SELESAI.")

# ===== CELL 2 =====
# ============================================================
# CELL 3A
# BUILD THE MODEL DATASET
# ============================================================

print("Starting Cell 3A...")

# Copy the genesis valuesset
genesis_model = genesis_domain.copy()

# ------------------------------------------------------------
# Unit conversion
# ------------------------------------------------------------

# SST: Kelvin -> Celsius
genesis_model["SST_C"] = (
    genesis_model["era5_sst"] - 273.15
)

# Total precipitation: meter -> mm
genesis_model["TP_mm"] = (
    genesis_model["era5_tp"] * 1000
)

# Relative vorticity: s^-1 -> x10^-5 s^-1
genesis_model["VORT_x1e5"] = (
    genesis_model["era5_vo"] * 1e5
)


# ------------------------------------------------------------
# Build the 7 model variables
# ------------------------------------------------------------

X = pd.DataFrame({
    "Latitude":
        pd.to_numeric(
            genesis_model["LAT"],
            errors="coerce"
        ),

    "SST":
        pd.to_numeric(
            genesis_model["SST_C"],
            errors="coerce"
        ),

    "Total_Precipitation":
        pd.to_numeric(
            genesis_model["TP_mm"],
            errors="coerce"
        ),

    "Vertical_Velocity":
        pd.to_numeric(
            genesis_model["era5_w"],
            errors="coerce"
        ),

    "Relative_Vorticity":
        pd.to_numeric(
            genesis_model["VORT_x1e5"],
            errors="coerce"
        ),

    "Wind":
        pd.to_numeric(
            genesis_model["WIND_km/h"],
            errors="coerce"
        ),

    "Pressure":
        pd.to_numeric(
            genesis_model["PRESSURE_hPa"],
            errors="coerce"
        )
})


print("=" * 60)
print("CELL 3A RESULTS")
print("=" * 60)

print("Number of samples :", len(X))
print("Number of variables :", len(X.columns))
print("Shape of X :", X.shape)

print("\nVariable names:")
print(X.columns.tolist())

print("\nFirst 5 rows:")
print(X.head())

print("\nMissing values:")
print(X.isna().sum())

print("\nCELL 3A SELESAI.")

# ===== CELL 3 =====
# ============================================================
# CELL 3B
# DESCRIPTIVE STATISTICS
# ============================================================

print("Starting Cell 3B...")

stats = pd.DataFrame({
    "N": X.count(),
    "Mean": X.mean(),
    "Median": X.median(),
    "Std": X.std(),
    "Min": X.min(),
    "Max": X.max(),
    "Skewness": X.skew()
})

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(stats.round(4))

print("\nCELL 3B SELESAI.")

# ===== CELL 4 =====
# ============================================================
# CELL 3C
# CORRELATION
# ============================================================

print("Starting Cell 3C...")


# ------------------------------------------------------------
# Pearson
# ------------------------------------------------------------

pearson_corr = X.corr(
    method="pearson"
)

print("=" * 60)
print("CORRELATION PEARSON")
print("=" * 60)

print(
    pearson_corr.round(3)
)


# ------------------------------------------------------------
# Spearman
# ------------------------------------------------------------

spearman_corr = X.corr(
    method="spearman"
)

print("\n" + "=" * 60)
print("CORRELATION SPEARMAN")
print("=" * 60)

print(
    spearman_corr.round(3)
)


print("\nCELL 3C SELESAI.")

# ===== CELL 5 =====
# ============================================================
# CELL 3D
# EMPIRICAL CDF TRANSFORMATION
# ============================================================

print("Starting Cell 3D...")

n = len(X)

U = pd.DataFrame(
    index=X.index
)

for col in X.columns:

    # Ranking using pandas
    ranks = X[col].rank(
        method="average"
    )

    # Mid-rank empirical CDF
    U[col] = (
        ranks - 0.5
    ) / n


print("=" * 60)
print("ECDF RESULTS")
print("=" * 60)

print("Shape of U :", U.shape)

print("\nMinimum value of each variable:")
print(U.min())

print("\nMaximum value of each variable:")
print(U.max())

print("\nFirst 5 rows:")
print(U.head())

print("\nCELL 3D SELESAI.")

# ===== CELL 6 =====
# ============================================================
# CELL 3E
# ECDF -> STANDARD NORMAL DISTRIBUTION TRANSFORMATION
# ============================================================

from statistics import NormalDist

print("Starting Cell 3E...")

normal_dist = NormalDist()

Z = pd.DataFrame(
    index=U.index
)

for col in U.columns:

    Z[col] = U[col].apply(
        normal_dist.inv_cdf
    )


print("=" * 60)
print("GAUSSIAN TRANSFORMATION RESULTS")
print("=" * 60)

print("Shape of Z :", Z.shape)

print("\nMean:")
print(Z.mean().round(4))

print("\nStandard deviation:")
print(Z.std().round(4))

print("\nFirst 5 rows:")
print(Z.head())

print("\nCELL 3E SELESAI.")

# ===== CELL 7 =====
# ============================================================
# CELL 3F-REVISI
# GAUSSIAN COPULA WITH COMPLETE DATA
# ============================================================

print("=" * 70)
print("REBUILDING THE GAUSSIAN COPULA")
print("=" * 70)

# ------------------------------------------------------------
# 1. Take only rows that are complete for all variables
# ------------------------------------------------------------

X_complete = X.dropna().copy()

print("Total number of events :", len(X))
print("Number of complete rows :", len(X_complete))
print("Excluded rows :", len(X) - len(X_complete))


# ------------------------------------------------------------
# 2. Display excluded events
# ------------------------------------------------------------

excluded_index = X.index.difference(
    X_complete.index
)

print("\nNumber of incomplete events:")
print(len(excluded_index))


if len(excluded_index) > 0:

    print("\nExcluded events:")

    print(
        genesis_model.loc[
            excluded_index,
            [
                "SID",
                "NAME",
                "SEASON",
                "LAT",
                "LON",
                "ZONE"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 3. ECDF
# ------------------------------------------------------------

n = len(X_complete)

U_complete = pd.DataFrame(
    index=X_complete.index
)

for col in X_complete.columns:

    ranks = X_complete[col].rank(
        method="average"
    )

    U_complete[col] = (
        ranks - 0.5
    ) / n


# ------------------------------------------------------------
# 4. Gaussian transformation
# ------------------------------------------------------------

from statistics import NormalDist

normal_dist = NormalDist()

Z_complete = pd.DataFrame(
    index=U_complete.index
)

for col in U_complete.columns:

    Z_complete[col] = [
        normal_dist.inv_cdf(value)
        for value in U_complete[col]
    ]


# ------------------------------------------------------------
# 5. Gaussian Copula Matrix
# ------------------------------------------------------------

copula_corr = Z_complete.corr(
    method="pearson"
)


print("\n" + "=" * 70)
print("MATRIKS CORRELATION GAUSSIAN COPULA")
print("=" * 70)

print(
    copula_corr.round(4)
)


# ------------------------------------------------------------
# 6. Check missing values
# ------------------------------------------------------------

print("\nNumber of missing values:")
print(
    copula_corr.isna().sum().sum()
)


# ------------------------------------------------------------
# 7. Simple Cholesky
# ------------------------------------------------------------

def cholesky_python(matrix):

    n = len(matrix)

    L = [
        [0.0 for _ in range(n)]
        for _ in range(n)
    ]

    for i in range(n):

        for j in range(i + 1):

            total = 0.0

            for k in range(j):

                total += (
                    L[i][k] *
                    L[j][k]
                )

            if i == j:

                value = (
                    matrix[i][i]
                    - total
                )

                if value <= 0:

                    return False, None

                L[i][j] = value ** 0.5

            else:

                if L[j][j] == 0:

                    return False, None

                L[i][j] = (
                    matrix[i][j]
                    - total
                ) / L[j][j]

    return True, L


is_pd, L = cholesky_python(
    copula_corr.values.tolist()
)


print("\n" + "=" * 70)
print("MATRIX CHECK")
print("=" * 70)

if is_pd:

    print(
        "✓ Gaussian Copula Matrix "
        "positive definite."
    )

else:

    print(
        "⚠ Gaussian Copula Matrix "
        "is not positive definite."
    )


# ------------------------------------------------------------
# 8. Ensure VAMEI and SENYAR are still included
# ------------------------------------------------------------

target = genesis_model[
    genesis_model["NAME"]
    .astype(str)
    .str.upper()
    .isin(["VAMEI", "SENYAR"])
]

print("\n" + "=" * 70)
print("VAMEI DAN SENYAR")
print("=" * 70)

print(
    target[
        [
            "SID",
            "NAME",
            "SEASON",
            "LAT",
            "LON",
            "ZONE"
        ]
    ].to_string(index=False)
)

print("\nCELL 3F-REVISED COMPLETED.")

# ===== CELL 8 =====
# ============================================================
# CELL 3F
# MATRIKS CORRELATION GAUSSIAN COPULA
# SAFE VERSION — WITHOUT EIGENVALUE
# ============================================================

print("Starting Cell 3F...")


# ------------------------------------------------------------
# 1. HITUNG CORRELATION PADA RUANG GAUSSIAN
# ------------------------------------------------------------

copula_corr = Z.corr(
    method="pearson"
)


# ------------------------------------------------------------
# 2. DISPLAY THE RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MATRIKS CORRELATION GAUSSIAN COPULA")
print("=" * 60)

print(
    copula_corr.round(4)
)


# ------------------------------------------------------------
# 3. CHECK SHAPE
# ------------------------------------------------------------

print("\nMatrix shape:")
print(copula_corr.shape)


# ------------------------------------------------------------
# 4. CHECK DIAGONAL
# ------------------------------------------------------------

print("\nDiagonal values:")

print(
    copula_corr.values[
        range(len(copula_corr)),
        range(len(copula_corr))
    ]
)


# ------------------------------------------------------------
# 5. CHECK SYMMETRY
# ------------------------------------------------------------

is_symmetric = (
    copula_corr
    .equals(copula_corr.T)
)

print(
    "\nSymmetric matrix:",
    is_symmetric
)


# ------------------------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------------------------

print(
    "\nNumber of missing values:",
    copula_corr.isna().sum().sum()
)


print("\nCELL 3F SELESAI.")

# ===== CELL 9 =====
# ============================================================
# CELL 3G
# SUBSET SIMULATION PREPARATION
# ============================================================

import numpy as np
import math
from statistics import NormalDist

print("Preparing Subset Simulation...")


# ============================================================
# 1. CONFIGURATION
# ============================================================

N = 20000

p0 = 0.10

sigma = 1.0

random_seed = 20261002

rng = np.random.default_rng(
    random_seed
)

normal_dist = NormalDist()


# ============================================================
# 2. CHOLESKY MATRIX
# ============================================================

is_pd, L_list = cholesky_python(
    copula_corr.values.tolist()
)

if not is_pd:

    raise ValueError(
        "Gaussian Copula Matrix is not positive definite."
    )

L = np.array(
    L_list,
    dtype=float
)


# ============================================================
# 3. SORTED MARGINALS
# ============================================================

sorted_marginals = {}

for col in X_complete.columns:

    sorted_marginals[col] = np.sort(
        X_complete[col].to_numpy()
    )


# ============================================================
# 4. NORMAL CDF FUNCTION
# ============================================================

def normal_cdf(z):

    return 0.5 * (
        1.0 +
        math.erf(
            z / math.sqrt(2.0)
        )
    )


# ============================================================
# 5. EMPIRICAL QUANTILE
# ============================================================

def empirical_ppf(
    u,
    sorted_values
):

    n = len(sorted_values)

    index = math.ceil(
        u * n
    ) - 1

    if index < 0:

        index = 0

    if index >= n:

        index = n - 1

    return float(
        sorted_values[index]
    )


# ============================================================
# 6. LATITUDE TRANSFORMATION
# ============================================================

lat_sorted = sorted_marginals[
    "Latitude"
]


def latitude_from_z(
    z_lat
):

    u = normal_cdf(
        z_lat
    )

    return empirical_ppf(
        u,
        lat_sorted
    )


# ============================================================
# 7. EVENT FUNCTION
# ============================================================

def performance_function(
    y
):

    # y = independent standard normal space

    z = L.dot(y)

    latitude = latitude_from_z(
        z[0]
    )

    return abs(latitude) - 5.0


# ============================================================
# 8. CEK FUNGSI
# ============================================================

print("\nNumber of variables :", len(X_complete.columns))

print(
    "Number of samples   :", N
)

print(
    "p0              :", p0
)

print(
    "MCMC sigma          :", sigma
)

print(
    "Random seed         :", random_seed
)

print("\nPreparation completed.")

# ===== CELL 10 =====
# ============================================================
# CELL 3H
# SUBSET SIMULATION
# ============================================================

print("=" * 70)
print("STARTING SUBSET SIMULATION")
print("=" * 70)


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

n_dim = len(X_complete.columns)

n_seed = int(
    round(N * p0)
)

chain_length = int(
    round(1 / p0)
)


# ------------------------------------------------------------
# LEVEL 0
# ------------------------------------------------------------

Y = rng.normal(
    size=(N, n_dim)
)

G = np.array([
    performance_function(y)
    for y in Y
])


levels = []

probability_factor = 1.0

level = 0


print("\nLEVEL 0")
print(
    "Number of samples :",
    N
)


# ============================================================
# LOOP SUBSET SIMULATION
# ============================================================

while True:

    # --------------------------------------------------------
    # Sort by the smallest G values
    # --------------------------------------------------------

    order = np.argsort(
        G,
        kind="stable"
    )

    G_sorted = G[
        order
    ]


    # --------------------------------------------------------
    # Check whether the 10% subset already reaches the rare event
    # --------------------------------------------------------

    if G_sorted[
        n_seed - 1
    ] <= 0:

        final_count = int(
            np.sum(G <= 0)
        )

        final_conditional_probability = (
            final_count / N
        )

        final_probability = (
            probability_factor
            * final_conditional_probability
        )


        levels.append({

            "Level": level,

            "Threshold_G": 0.0,

            "Latitude_Boundary": 5.0,

            "Number_of_Events": final_count,

            "Conditional_Probability":
                final_conditional_probability,

            "Cumulative_Probability":
                final_probability,

            "Acceptance_Rate":
                np.nan,

            "Status":
                "FINAL"

        })


        # ====================================================
        # SAVE FINAL POPULATION
        # ====================================================

        # All 20,000 samples in the last level
        Y_level_final = Y.copy()

        # Performance function values of all samples
        G_level_final = G.copy()

        # Mask of samples that truly satisfy the rare event
        mask_rare_final = (
            G_level_final <= 0
        )

        # Latent Gaussian samples that satisfy |latitude| <= 5°
        Y_rare_final = Y_level_final[
            mask_rare_final
        ].copy()

        print("\nFinal samples were successfully saved.")
        print(
            "Total final samples :",
            len(Y_level_final)
        )
        print(
            "Number of rare-event samples :",
            len(Y_rare_final)
        )

        break


    # --------------------------------------------------------
    # Subset threshold
    # --------------------------------------------------------

    threshold = float(
        G_sorted[
            n_seed - 1
        ]
    )


    latitude_boundary = (
        5.0 + threshold
    )


    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    seeds = Y[
        order[
            :n_seed
        ]
    ].copy()


    # --------------------------------------------------------
    # MCMC
    # --------------------------------------------------------

    new_Y = np.empty_like(
        Y
    )

    accepted = 0

    proposed = 0

    current_index = 0


    for seed_index in range(
        n_seed
    ):

        current = seeds[
            seed_index
        ].copy()


        # Save the seed
        new_Y[
            current_index
        ] = current

        current_index += 1


        # ----------------------------------------------------
        # Generate the chain
        # ----------------------------------------------------

        for step in range(
            1,
            chain_length
        ):

            candidate = current.copy()


            # Randomly choose one component
            j = int(
                rng.integers(
                    n_dim
                )
            )


            candidate[
                j
            ] += (
                sigma *
                rng.normal()
            )


            candidate_G = (
                performance_function(
                    candidate
                )
            )


            proposed += 1


            # ------------------------------------------------
            # Candidate must remain inside the subset
            # ------------------------------------------------

            if candidate_G <= threshold:

                current_norm = np.dot(
                    current,
                    current
                )

                candidate_norm = np.dot(
                    candidate,
                    candidate
                )


                log_alpha = -0.5 * (
                    candidate_norm
                    - current_norm
                )


                random_value = max(
                    rng.random(),
                    1e-15
                )


                if (
                    log_alpha >= 0
                    or
                    math.log(
                        random_value
                    ) < log_alpha
                ):

                    current = candidate

                    accepted += 1


            # Save sample
            new_Y[
                current_index
            ] = current

            current_index += 1


    # --------------------------------------------------------
    # Population of the next level
    # --------------------------------------------------------

    Y = new_Y


    G = np.array([
        performance_function(y)
        for y in Y
    ])


    # --------------------------------------------------------
    # Update probability
    # --------------------------------------------------------

    probability_factor *= p0

    level += 1


    # --------------------------------------------------------
    # Acceptance rate
    # --------------------------------------------------------

    acceptance_rate = (
        accepted / proposed
        if proposed > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Record the results
    # --------------------------------------------------------

    levels.append({

        "Level": level,

        "Threshold_G": threshold,

        "Latitude_Boundary":
            latitude_boundary,

        "Number_of_Events":
            int(np.sum(G <= 0)),

        "Conditional_Probability":
            p0,

        "Cumulative_Probability":
            np.nan,

        "Acceptance_Rate":
            acceptance_rate,

        "Status":
            "INTERMEDIATE"

    })


    # --------------------------------------------------------
    # Show progress
    # --------------------------------------------------------

    print(
        f"\nLevel {level}"
    )

    print(
        f"Threshold G      : {threshold:.4f}"
    )

    print(
        f"Latitude boundary:  ±{latitude_boundary:.2f}°"
    )

    print(
        f"Equatorial events:  {int(np.sum(G <= 0))}"
    )

    print(
        f"Acceptance rate  : {acceptance_rate:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

results_ss = pd.DataFrame(
    levels
)


print("\n" + "=" * 70)
print("FINAL RESULTS SUBSET SIMULATION")
print("=" * 70)

print(
    results_ss.to_string(
        index=False
    )
)


print("\nRare-event Probability:")

print(
    f"{final_probability:.8f}"
)

print(
    f"{final_probability * 100:.4f}%"
)


print("\nCELL 3H SELESAI.")

# ===== CELL 11 =====
# ====================================================
# SAVE FINAL POPULATION
# ====================================================

Y_level_final = Y.copy()
G_level_final = G.copy()

mask_rare_final = (
    G_level_final <= 0
)

Y_rare_final = Y_level_final[
    mask_rare_final
].copy()

print("\nFinal samples were successfully saved.")
print(
    "Total final samples :",
    len(Y_level_final)
)
print(
    "Number of rare-event samples :",
    len(Y_rare_final)
)

# ----------------------------------------------------
# SAVE TO FILE AS WELL
# So the values are not lost if the kernel crashes again
# ----------------------------------------------------

np.savez_compressed(
    "subset_simulation_final.npz",
    Y_level_final=Y_level_final,
    G_level_final=G_level_final,
    Y_rare_final=Y_rare_final,
    L=L
)

X_complete.to_csv(
    "X_complete_model.csv",
    index=False
)

print(
    "\n✓ Backup file subset_simulation_final.npz created"
)
print(
    "✓ Backup file X_complete_model.csv created"
)

# ===== CELL 12 =====
# ============================================================
# CELL 3I-A
# CHECK THE FINAL SUBSET SIMULATION SAMPLES
# ============================================================

print("=" * 75)
print("CHECKING THE FINAL SUBSET SIMULATION POPULATION")
print("=" * 75)

print(
    "Shape of Y_level_final :",
    Y_level_final.shape
)

print(
    "Shape of G_level_final :",
    G_level_final.shape
)

print(
    "Shape of Y_rare_final  :",
    Y_rare_final.shape
)

print(
    "\nNumber of G <= 0       :",
    np.sum(
        G_level_final <= 0
    )
)

print(
    "Final proportion       :",
    np.mean(
        G_level_final <= 0
    )
)

print(
    "SS Probability         :",
    p0 * p0 *
    np.mean(
        G_level_final <= 0
    )
)

print(
    "SS Probability (%)     :",
    100 *
    p0 * p0 *
    np.mean(
        G_level_final <= 0
    )
)

assert (
    Y_level_final.shape[0]
    == N
), "The final number of samples is not N"

assert (
    len(Y_rare_final)
    == final_count
), "The number of rare events does not match final_count"

print("\n✓ Check completed.")

# ===== CELL 13 =====
# ============================================================
# CELL 3I-B
# VIEW THE DEFINITION OF performance_function
# ============================================================

import inspect

print("=" * 75)
print("DEFINITION OF performance_function")
print("=" * 75)

try:
    source_pf = inspect.getsource(
        performance_function
    )

    print(source_pf)

except Exception as e:

    print(
        "inspect.getsource was not successful:",
        e
    )

    print(
        "\nNames of objects/functions used "
        "by performance_function:"
    )

    print(
        performance_function.__code__.co_names
    )

print("\n" + "=" * 75)

# ===== CELL 14 =====
# ============================================================
# CELL 3I-C LITE
# TRANSFORMATION OF RARE-EVENT SAMPLES
# LIGHT VERSION - WITHOUT SCIPY
# ============================================================

import numpy as np
import pandas as pd
import math
from statistics import NormalDist

print("=" * 80)
print("RARE-EVENT TRANSFORMATION - LITE MODE")
print("=" * 80)


# ============================================================
# 1. LOAD BACKUP
# ============================================================

print("\n[1/7] Reading backup files...")

backup = np.load(
    "subset_simulation_final.npz",
    allow_pickle=False
)

Y_rare = backup["Y_rare_final"]
L_safe = backup["L"]

X_obs = pd.read_csv(
    "X_complete_model.csv"
)

print("    Y_rare shape :", Y_rare.shape)
print("    L shape      :", L_safe.shape)
print("    X_obs shape  :", X_obs.shape)


# ============================================================
# 2. VALIDASI UKURAN
# ============================================================

print("\n[2/7] Checking shapes...")

if Y_rare.ndim != 2:
    raise ValueError("Y_rare is not a 2-dimensional array.")

if Y_rare.shape[1] != 7:
    raise ValueError(
        f"Number of variables = {Y_rare.shape[1]}, it should be 7."
    )

if L_safe.shape != (7, 7):
    raise ValueError(
        f"Shape of L = {L_safe.shape}, it should be (7, 7)."
    )

print("    Shape validation OK")


# ============================================================
# 3. UJI SATU SAMPEL DULU
# ============================================================

print("\n[3/7] Testing the transformation on one sample...")

y_test = Y_rare[0]

# Sama seperti performance_function:
z_test = L_safe.dot(y_test)

print("    first y :", y_test)
print("    first z :", z_test)

print("    Single-sample transformation OK")


# ============================================================
# 4. NORMAL CDF FUNCTION TANPA SCIPY
# ============================================================

normal_std = NormalDist(
    mu=0.0,
    sigma=1.0
)

def normal_cdf_safe(x):
    return normal_std.cdf(
        float(x)
    )


# ============================================================
# 5. LIGHTWEIGHT INVERSE EMPIRICAL QUANTILE FUNCTION
# ============================================================

def inverse_empirical_light(
    sorted_values,
    u
):
    """
    Linear interpolation of the empirical quantile.

    sorted_values : sorted observed values
    u             : probability between 0 and 1
    """

    n = len(sorted_values)

    if n == 1:
        return float(
            sorted_values[0]
        )

    u = min(
        max(float(u), 0.0),
        1.0
    )

    position = (
        u * (n - 1)
    )

    lower = int(
        math.floor(position)
    )

    upper = int(
        math.ceil(position)
    )

    if lower == upper:
        return float(
            sorted_values[lower]
        )

    weight = (
        position - lower
    )

    return float(
        sorted_values[lower]
        +
        weight
        *
        (
            sorted_values[upper]
            -
            sorted_values[lower]
        )
    )


# ============================================================
# 6. PREPARE OBSERVATIONAL DATA
# ============================================================

print("\n[4/7] Preparing empirical marginals...")

columns = list(
    X_obs.columns
)

print("\nModel column order:")

for i, col in enumerate(columns):
    print(
        f"    {i}: {col}"
    )


# meteorological variables only, latitude does not need to be transformed
meteo_columns = columns[1:]

sorted_obs = {}

for col in meteo_columns:

    values = pd.to_numeric(
        X_obs[col],
        errors="coerce"
    ).dropna().to_numpy(
        dtype=float
    )

    values.sort()

    sorted_obs[col] = values

    print(
        f"    {col}: {len(values)} values"
    )


# ============================================================
# 7. TRANSFORM STEP BY STEP
# ============================================================

print("\n[5/7] Transforming 10,297 samples...")
print("      Please wait, the process is performed gradually.")

n = Y_rare.shape[0]

results = {
    col: np.empty(
        n,
        dtype=float
    )
    for col in meteo_columns
}


# Do not store the full Z and U arrays.
# One sample is processed and then discarded.
for i in range(n):

    # exactly the same as in performance_function
    z = L_safe.dot(
        Y_rare[i]
    )

    # column 0 = latitude
    # columns 1-6 = meteorological variables
    for j, col in enumerate(
        meteo_columns,
        start=1
    ):

        u = normal_cdf_safe(
            z[j]
        )

        results[col][i] = (
            inverse_empirical_light(
                sorted_obs[col],
                u
            )
        )

    # Progress every 1000 samples
    if (
        (i + 1) % 1000 == 0
        or
        i == n - 1
    ):
        print(
            f"      completed {i + 1:,}/{n:,}"
        )


# ============================================================
# RESULT DATAFRAME
# ============================================================

print("\n[6/7] Building DataFrame...")

X_rare_meteo = pd.DataFrame(
    results
)


print(
    "    Shape:",
    X_rare_meteo.shape
)

print("\nFirst 5 samples:")

print(
    X_rare_meteo.head()
)


# ============================================================
# SAVE DIRECTLY
# ============================================================

print("\n[7/7] Menyimpan results...")

X_rare_meteo.to_csv(
    "rare_event_meteorology_final.csv",
    index=False
)

print(
    "    File: rare_event_meteorology_final.csv"
)

print("\n" + "=" * 80)
print("CELL 3I-C LITE COMPLETED")
print("=" * 80)

# ===== CELL 15 =====
# ============================================================
# CELL 3I-D
# TABLE 3.13 STATISTICS
# ============================================================

import pandas as pd
import numpy as np

df_rare = pd.read_csv(
    "rare_event_meteorology_final.csv"
)

print(
    "Number of samples:",
    len(df_rare)
)

print(
    "Columns:",
    df_rare.columns.tolist()
)


# ------------------------------------------------------------
# Detect the order of the six parameters
# ------------------------------------------------------------

cols = list(
    df_rare.columns
)

if len(cols) != 6:
    raise ValueError(
        "There should be 6 meteorological parameters."
    )


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

records = []

for col in cols:

    x = pd.to_numeric(
        df_rare[col],
        errors="coerce"
    ).dropna()

    records.append({

        "Parameters":
            col,

        "Median":
            x.quantile(0.50),

        "Q1":
            x.quantile(0.25),

        "Q3":
            x.quantile(0.75),

        "P5":
            x.quantile(0.05),

        "P95":
            x.quantile(0.95),

        "Minimum":
            x.min(),

        "Maximum":
            x.max()
    })


stats_rare = pd.DataFrame(
    records
)

print("\nSTATISTIK RARE EVENT:")

print(
    stats_rare.round(5).to_string(
        index=False
    )
)

