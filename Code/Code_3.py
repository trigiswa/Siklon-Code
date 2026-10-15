# ============================================================
# CHAPTER III - 3.1.2.3
# CHARACTERISTICS OF EACH CLUSTER
#
# Method:
# 1. Read the tropical cyclone dataset
# 2. Read the optimal clustering output from Section 3.1.2.2
# 3. Compute trajectory characteristics for each SID
# 4. Summarize the characteristics of each cluster
# 5. Plot one trajectory-pattern map for each cluster
#
# Output:
# - Figure 3.7: one plot for each cluster trajectory pattern
# - Table 3.5: main characteristics of each cluster
# - Additional supporting tables in Excel
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.gridliner import (
    LONGITUDE_FORMATTER,
    LATITUDE_FORMATTER
)

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ============================================================
# 1. FILE SETTINGS
# ============================================================

DATA_FILE = "Siklon_Tropis_ERA5_Extracted.xlsx"

CLUSTER_FILE = os.path.join(
    "output_3_1_2",
    "Clustering_Results_3_1_2_2.xlsx"
)

OUTPUT_DIR = "output_3_1_2"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_EXCEL = os.path.join(
    OUTPUT_DIR,
    "Cluster_Characteristics_3_1_2_3.xlsx"
)

FIGURE_TEMPLATE = os.path.join(
    OUTPUT_DIR,
    "Figure_3.7_Cluster_{cluster_id}_Trajectory_Pattern.png"
)


# ============================================================
# 2. GENESIS SELECTION DOMAIN
# ============================================================

LON_MIN = 95
LON_MAX = 145

LAT_MIN = -15
LAT_MAX = 25

# Equatorial zone boundaries
EQ_MIN = -5
EQ_MAX = 5


# ============================================================
# 3. PLOT DISPLAY DOMAIN
# ============================================================
# This plotting extent is wider than the research domain so
# that post-genesis trajectories remain visible.
# ============================================================

PLOT_LON_MIN = 80
PLOT_LON_MAX = 170

PLOT_LAT_MIN = -35
PLOT_LAT_MAX = 45


# ============================================================
# 4. CLUSTER COLORS
# ============================================================

CLUSTER_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c"
}


# ============================================================
# 5. READ MAIN DATA
# ============================================================

SHEET_NAME = "Sheet1"

df = pd.read_excel(
    DATA_FILE,
    sheet_name=SHEET_NAME
)


# ============================================================
# 6. DATA CLEANING
# ============================================================

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

df["LAT"] = pd.to_numeric(
    df["LAT"],
    errors="coerce"
)

df["LON"] = pd.to_numeric(
    df["LON"],
    errors="coerce"
)

df["SEASON"] = pd.to_numeric(
    df["SEASON"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "SID",
        "ISO_TIME",
        "LAT",
        "LON"
    ]
).copy()

df = df[
    df["LAT"].between(
        -90,
        90
    )
].copy()

df = df[
    df["LON"].between(
        -180,
        360
    )
].copy()

df = df.drop_duplicates(
    subset=[
        "SID",
        "ISO_TIME"
    ],
    keep="first"
)

df = df.sort_values(
    [
        "SID",
        "ISO_TIME"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 7. READ CLUSTERING RESULTS
# ============================================================

if not os.path.exists(
    CLUSTER_FILE
):

    raise FileNotFoundError(
        "The clustering results file was not found.\n"
        "Please run Section 3.1.2.2 first.\n"
        f"Missing file: {CLUSTER_FILE}"
    )

assignment = pd.read_excel(
    CLUSTER_FILE,
    sheet_name="Cluster Assignments",
    dtype={
        "SID": str
    }
)

medoid_df = pd.read_excel(
    CLUSTER_FILE,
    sheet_name="Cluster Medoids",
    dtype={
        "Medoid SID": str
    }
)

# Make sure the main SID column is also string

df["SID"] = (
    df["SID"]
    .astype(str)
)


# ============================================================
# 8. SELECT TRACKS INCLUDED IN THE CLUSTERING RESULTS
# ============================================================

selected_sids = (
    assignment["SID"]
    .astype(str)
    .unique()
)

track = df[
    df["SID"].isin(
        selected_sids
    )
].copy()

print("=" * 70)
print("DATA FOR SECTION 3.1.2.3")
print("=" * 70)

print(
    "Number of SID:",
    assignment["SID"].nunique()
)

print(
    "Number of clusters:",
    assignment["Cluster"].nunique()
)

print(
    "\nNumber of members in each cluster:"
)

print(
    assignment["Cluster"]
    .value_counts()
    .sort_index()
)

print("=" * 70)


# ============================================================
# 9. HAVERSINE FUNCTION
# ============================================================

EARTH_RADIUS_KM = 6371.0088


def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(
        lat1
    )

    lon1 = np.radians(
        lon1
    )

    lat2 = np.radians(
        lat2
    )

    lon2 = np.radians(
        lon2
    )

    dlat = (
        lat2 - lat1
    )

    dlon = (
        lon2 - lon1
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        +
        np.cos(
            lat1
        )
        *
        np.cos(
            lat2
        )
        *
        np.sin(
            dlon / 2
        ) ** 2
    )

    c = (
        2
        *
        np.arctan2(
            np.sqrt(a),
            np.sqrt(
                1 - a
            )
        )
    )

    return (
        EARTH_RADIUS_KM
        *
        c
    )


# ============================================================
# 10. BEARING FUNCTION
# ============================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(
        lat1
    )

    lat2 = np.radians(
        lat2
    )

    delta_lon = np.radians(
        lon2 - lon1
    )

    x = (
        np.sin(
            delta_lon
        )
        *
        np.cos(
            lat2
        )
    )

    y = (
        np.cos(
            lat1
        )
        *
        np.sin(
            lat2
        )
        -
        np.sin(
            lat1
        )
        *
        np.cos(
            lat2
        )
        *
        np.cos(
            delta_lon
        )
    )

    bearing = np.degrees(
        np.arctan2(
            x,
            y
        )
    )

    return (
        bearing + 360
    ) % 360


# ============================================================
# 11. DIRECTION CLASSIFICATION
# ============================================================

def bearing_to_direction(
    bearing
):

    directions = [
        "North",
        "Northeast",
        "East",
        "Southeast",
        "South",
        "Southwest",
        "West",
        "Northwest"
    ]

    index = int(
        (
            (
                bearing + 22.5
            )
            %
            360
        )
        /
        45
    )

    return directions[
        index
    ]


# ============================================================
# 12. CALCULATE CHARACTERISTICS FOR EACH SID
# ============================================================

records = []

for sid, g in track.groupby(
    "SID"
):

    g = g.sort_values(
        "ISO_TIME"
    ).copy()

    lat = (
        g["LAT"]
        .to_numpy(
            dtype=float
        )
    )

    lon = (
        g["LON"]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Track length
    # --------------------------------------------------------

    if len(g) >= 2:

        segment_distance = haversine(
            lat[:-1],
            lon[:-1],
            lat[1:],
            lon[1:]
        )

        track_length = np.nansum(
            segment_distance
        )

    else:

        track_length = 0

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration_hours = (
        (
            g["ISO_TIME"].iloc[-1]
            -
            g["ISO_TIME"].iloc[0]
        )
        .total_seconds()
        /
        3600
    )

    # --------------------------------------------------------
    # Bearing: genesis -> termination
    # --------------------------------------------------------

    bearing = calculate_bearing(
        lat[0],
        lon[0],
        lat[-1],
        lon[-1]
    )

    direction = bearing_to_direction(
        bearing
    )

    records.append({
        "SID": sid,
        "Track_Length_km": track_length,
        "Duration_Hours": duration_hours,
        "Duration_Days": duration_hours / 24,
        "Termination_Lat": lat[-1],
        "Termination_Lon": lon[-1],
        "Bearing_deg": bearing,
        "Direction": direction
    })

trajectory_stats = pd.DataFrame(
    records
)


# ============================================================
# 13. MERGE WITH CLUSTER ASSIGNMENT
# ============================================================

cluster_data = assignment.merge(
    trajectory_stats,
    on="SID",
    how="left"
)


# ============================================================
# 14. ZONE DISTRIBUTION BY CLUSTER
# ============================================================

zone_distribution = (
    cluster_data
    .groupby(
        [
            "Cluster",
            "Genesis_Zone"
        ]
    )
    .size()
    .reset_index(
        name="Count"
    )
)

zone_distribution[
    "Percentage within Cluster (%)"
] = (
    zone_distribution[
        "Count"
    ]
    /
    zone_distribution[
        "Cluster"
    ].map(
        cluster_data[
            "Cluster"
        ]
        .value_counts()
    )
    *
    100
)


# ============================================================
# 15. DIRECTION DISTRIBUTION BY CLUSTER
# ============================================================

direction_distribution = (
    cluster_data
    .groupby(
        [
            "Cluster",
            "Direction"
        ]
    )
    .size()
    .reset_index(
        name="Count"
    )
)

direction_distribution[
    "Percentage within Cluster (%)"
] = (
    direction_distribution[
        "Count"
    ]
    /
    direction_distribution[
        "Cluster"
    ].map(
        cluster_data[
            "Cluster"
        ]
        .value_counts()
    )
    *
    100
)


# ============================================================
# 16. CREATE TABLE 3.5
# ============================================================

summary_records = []

for cluster_id in sorted(
    cluster_data["Cluster"].unique()
):

    g = cluster_data[
        cluster_data["Cluster"]
        ==
        cluster_id
    ].copy()

    # --------------------------------------------------------
    # Dominant genesis zone
    # --------------------------------------------------------

    zone_counts = (
        g["Genesis_Zone"]
        .value_counts()
    )

    dominant_zone = (
        zone_counts.index[0]
    )

    dominant_zone_n = int(
        zone_counts.iloc[0]
    )

    dominant_zone_pct = (
        dominant_zone_n
        /
        len(g)
        *
        100
    )

    # --------------------------------------------------------
    # Dominant direction
    # --------------------------------------------------------

    direction_counts = (
        g["Direction"]
        .value_counts()
    )

    dominant_direction = (
        direction_counts.index[0]
    )

    dominant_direction_n = int(
        direction_counts.iloc[0]
    )

    dominant_direction_pct = (
        dominant_direction_n
        /
        len(g)
        *
        100
    )

    # --------------------------------------------------------
    # Medoid
    # --------------------------------------------------------

    medoid_row = medoid_df[
        medoid_df[
            "Cluster"
        ]
        ==
        cluster_id
    ].iloc[0]

    medoid_name = (
        medoid_row[
            "Cyclone Name"
        ]
    )

    medoid_sid = (
        medoid_row[
            "Medoid SID"
        ]
    )

    # Find medoid season
    medoid_meta = assignment[
        assignment[
            "SID"
        ]
        ==
        medoid_sid
    ]

    if len(
        medoid_meta
    ) > 0:

        medoid_year = int(
            medoid_meta[
                "SEASON"
            ].iloc[0]
        )

    else:

        medoid_year = ""

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_records.append({
        "Cluster": cluster_id,
        "Number of Trajectories": len(g),
        "Percentage (%)": len(g) / len(cluster_data) * 100,
        "Dominant Genesis Zone": (
            f"{dominant_zone} "
            f"({dominant_zone_n}; "
            f"{dominant_zone_pct:.2f}%)"
        ),
        "Dominant Direction": (
            f"{dominant_direction} "
            f"({dominant_direction_n}; "
            f"{dominant_direction_pct:.2f}%)"
        ),
        "Mean Length (km)": g[
            "Track_Length_km"
        ].mean(),
        "Mean Duration (hours)": g[
            "Duration_Hours"
        ].mean(),
        "Mean Duration (days)": g[
            "Duration_Days"
        ].mean(),
        "Medoid": (
            f"{medoid_name} "
            f"({medoid_year})"
        )
    })

table_35 = pd.DataFrame(
    summary_records
)


# ============================================================
# 17. ADDITIONAL STATISTICS BY CLUSTER
# ============================================================

cluster_statistics = (
    cluster_data
    .groupby(
        "Cluster"
    )
    .agg(
        Number_of_Trajectories=(
            "SID",
            "count"
        ),
        Mean_Genesis_Latitude=(
            "Genesis_Lat",
            "mean"
        ),
        Mean_Genesis_Longitude=(
            "Genesis_Lon",
            "mean"
        ),
        Median_Genesis_Latitude=(
            "Genesis_Lat",
            "median"
        ),
        Median_Genesis_Longitude=(
            "Genesis_Lon",
            "median"
        ),
        Mean_Length_km=(
            "Track_Length_km",
            "mean"
        ),
        Median_Length_km=(
            "Track_Length_km",
            "median"
        ),
        Mean_Duration_Hours=(
            "Duration_Hours",
            "mean"
        ),
        Median_Duration_Hours=(
            "Duration_Hours",
            "median"
        ),
        Mean_Termination_Latitude=(
            "Termination_Lat",
            "mean"
        ),
        Mean_Termination_Longitude=(
            "Termination_Lon",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 18. SAVE RESULTS TO EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    table_35.round(
        2
    ).to_excel(
        writer,
        sheet_name="Table 3.5",
        index=False
    )

    cluster_statistics.round(
        2
    ).to_excel(
        writer,
        sheet_name="Cluster Statistics",
        index=False
    )

    zone_distribution.round(
        2
    ).to_excel(
        writer,
        sheet_name="Zone Distribution",
        index=False
    )

    direction_distribution.round(
        2
    ).to_excel(
        writer,
        sheet_name="Direction Distribution",
        index=False
    )

    medoid_df.to_excel(
        writer,
        sheet_name="Cluster Medoids",
        index=False
    )

    cluster_data.to_excel(
        writer,
        sheet_name="Data per SID",
        index=False
    )

print(
    "\nExcel successfully saved:"
)

print(
    OUTPUT_EXCEL
)


# ============================================================
# 19. FIGURE 3.7
# ONE TRAJECTORY-PATTERN MAP FOR EACH CLUSTER
# ============================================================

saved_figures = []

for cluster_id in sorted(
    cluster_data[
        "Cluster"
    ].unique()
):

    fig = plt.figure(
        figsize=(8.5, 7)
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    color = (
        CLUSTER_COLORS.get(
            cluster_id,
            "#1f77b4"
        )
    )

    cluster_members = cluster_data[
        cluster_data[
            "Cluster"
        ]
        ==
        cluster_id
    ]

    cluster_sids = (
        cluster_members[
            "SID"
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Extent
    # --------------------------------------------------------

    ax.set_extent(
        [
            PLOT_LON_MIN,
            PLOT_LON_MAX,
            PLOT_LAT_MIN,
            PLOT_LAT_MAX
        ],
        crs=ccrs.PlateCarree()
    )

    # --------------------------------------------------------
    # Basemap
    # --------------------------------------------------------

    ax.add_feature(
        cfeature.OCEAN.with_scale(
            "10m"
        ),
        facecolor="#dfe8ef",
        zorder=0
    )

    ax.add_feature(
        cfeature.LAND.with_scale(
            "10m"
        ),
        facecolor="#f2f2f2",
        zorder=0
    )

    ax.add_feature(
        cfeature.COASTLINE.with_scale(
            "10m"
        ),
        linewidth=0.65,
        edgecolor="black",
        zorder=4
    )

    ax.add_feature(
        cfeature.BORDERS.with_scale(
            "10m"
        ),
        linewidth=0.35,
        linestyle=":",
        edgecolor="gray",
        zorder=4
    )

    # --------------------------------------------------------
    # Gridlines
    # --------------------------------------------------------

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--"
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # --------------------------------------------------------
    # Zone boundaries
    # --------------------------------------------------------

    for lat_line, style in [
        (5, "--"),
        (0, "-."),
        (-5, "--")
    ]:

        ax.plot(
            [
                PLOT_LON_MIN,
                PLOT_LON_MAX
            ],
            [
                lat_line,
                lat_line
            ],
            color="gray",
            linestyle=style,
            linewidth=0.8,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    # --------------------------------------------------------
    # Genesis selection domain rectangle
    # --------------------------------------------------------

    domain_rectangle = Rectangle(
        (
            LON_MIN,
            LAT_MIN
        ),
        LON_MAX - LON_MIN,
        LAT_MAX - LAT_MIN,
        fill=False,
        edgecolor="black",
        linewidth=1.1,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=7
    )

    ax.add_patch(
        domain_rectangle
    )

    # --------------------------------------------------------
    # All trajectories belonging to this cluster
    # --------------------------------------------------------

    cluster_tracks = track[
        track[
            "SID"
        ].isin(
            cluster_sids
        )
    ]

    for sid, g in cluster_tracks.groupby(
        "SID"
    ):

        g = g.sort_values(
            "ISO_TIME"
        )

        ax.plot(
            g[
                "LON"
            ],
            g[
                "LAT"
            ],
            color=color,
            linewidth=0.75,
            alpha=0.30,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    # --------------------------------------------------------
    # Genesis points of cluster members
    # --------------------------------------------------------

    ax.scatter(
        cluster_members[
            "Genesis_Lon"
        ],
        cluster_members[
            "Genesis_Lat"
        ],
        s=16,
        color=color,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.80,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # --------------------------------------------------------
    # Cluster medoid (trajectory and genesis point only)
    # --------------------------------------------------------

    medoid_row = medoid_df[
        medoid_df[
            "Cluster"
        ]
        ==
        cluster_id
    ].iloc[0]

    medoid_sid = str(
        medoid_row[
            "Medoid SID"
        ]
    )

    medoid_track = track[
        track[
            "SID"
        ]
        ==
        medoid_sid
    ].sort_values(
        "ISO_TIME"
    )

    # White outline
    ax.plot(
        medoid_track[
            "LON"
        ],
        medoid_track[
            "LAT"
        ],
        color="white",
        linewidth=4.0,
        transform=ccrs.PlateCarree(),
        zorder=8
    )

    # Black medoid trajectory
    ax.plot(
        medoid_track[
            "LON"
        ],
        medoid_track[
            "LAT"
        ],
        color="black",
        linewidth=2.1,
        transform=ccrs.PlateCarree(),
        zorder=9
    )

    # Medoid genesis point
    ax.scatter(
        medoid_track[
            "LON"
        ].iloc[0],
        medoid_track[
            "LAT"
        ].iloc[0],
        marker="*",
        s=120,
        color="black",
        edgecolor="white",
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        (
            f"Cluster {cluster_id}\n"
            f"n = {len(cluster_members)}"
        ),
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7,
            label="Genesis"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.2,
            label="Cluster medoid"
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="white",
            markersize=12,
            label="Medoid genesis"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.1,
            label="Genesis selection domain"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=True,
        fontsize=9,
        columnspacing=1.8,
        handlelength=2.8,
        handletextpad=0.6,
        borderpad=0.6
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, top=0.90)

    output_figure = FIGURE_TEMPLATE.format(
        cluster_id=cluster_id
    )

    plt.savefig(
        output_figure,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    plt.close(fig)

    saved_figures.append(
        output_figure
    )

    print(
        f"\nFigure 3.7 - Cluster {cluster_id} saved:"
    )
    print(
        output_figure
    )


# ============================================================
# 20. PRINT TABLE 3.5
# ============================================================

print("\n" + "=" * 80)
print(
    "TABLE 3.5 MAIN CHARACTERISTICS OF EACH CLUSTER"
)
print("=" * 80)

print(
    table_35
    .round(2)
    .to_string(
        index=False
    )
)

print("=" * 80)

print("\nSaved figure files:")
for fig_path in saved_figures:
    print(fig_path)
