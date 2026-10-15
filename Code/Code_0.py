# ===== CELL 0 =====
import glob
import os
import zipfile
from tqdm import tqdm

# ==========================================
# PATH CONFIGURATION
# ==========================================
ERA5_BASE_DIR = r"C:\Users\LENOVO\Documents\Tugaslah Pokoknya\Draft\Data ERA5"

print("Starting file inspection and extraction...")

# Search all files inside the ERA5 folder (including year subfolders)
all_files = glob.glob(os.path.join(ERA5_BASE_DIR, "**", "*"), recursive=True)

extracted_count = 0
skipped_count = 0

for filepath in tqdm(all_files, desc="Extracting Files"):
  # Skip folders, process files only
  if os.path.isdir(filepath):
    continue

  # Check whether the file is a ZIP file (based on Magic Bytes 'PK\x03\x04')
  try:
    with open(filepath, "rb") as f:
      header = f.read(4)

    if header.startswith(b"PK\x03\x04"):
      target_dir = os.path.dirname(filepath)
      filename_without_ext = os.path.splitext(os.path.basename(filepath))[0]

      # Create a unique name/subfolder if needed so files are not overwritten
      with zipfile.ZipFile(filepath, "r") as z:
        for member in z.infolist():
          # Extract the file and add the original filename as a prefix
          extracted_filename = f"{filename_without_ext}_{member.filename}"
          extracted_path = os.path.join(target_dir, extracted_filename)

          # Extract only if the file does not already exist
          if not os.path.exists(extracted_path):
            with (
                z.open(member) as source,
                open(extracted_path, "wb") as target,
            ):
              target.write(source.read())

      # Remove or rename the old .nc file (which is actually a zip)
      # Here, the original zip file is renamed to use the .zip extension
      if not filepath.endswith(".zip"):
        os.rename(filepath, filepath + ".zip")

      extracted_count += 1
    else:
      skipped_count += 1

  except Exception as e:
    print(f"\nFailed to extract {filepath}: {e}")

print("\n=== EXTRACTION PROCESS COMPLETED ===")
print(f"Total file ZIP berresults diekstrak : {extracted_count}")
print(f"Total other / non-ZIP files            : {skipped_count}")
print("\nAll original NetCDF (.nc) files are now ready to be read!")

# ===== CELL 1 =====
import glob
import os
import warnings
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Ignore internal warnings from xarray/pandas to keep terminal output clean
warnings.filterwarnings("ignore")

# ==========================================
# 1. PATH CONFIGURATION FILE & FOLDER
# ==========================================
EXCEL_PATH = "Siklon Tropis.xlsx"
ERA5_BASE_DIR = r"C:\Users\LENOVO\Documents\Tugaslah Pokoknya\Draft\Data ERA5"
OUTPUT_PATH = "Siklon_Tropis_ERA5_Extracted.xlsx"

# ==========================================
# 2. READ AND PREPARE TROPICAL CYCLONE DATA
# ==========================================
print("Reading cyclone data from Excel...")
df = pd.read_excel(EXCEL_PATH)

# Ensure the ISO_TIME column is in datetime format
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
df["YEAR"] = df["ISO_TIME"].dt.year

extracted_df_list = []

# ==========================================
# 3. EXTRACT ERA5 DATA YEAR BY YEAR
# ==========================================
print("\nStarting the ERA5 variable extraction based on LAT, LON, and ISO_TIME...")

for year, group in df.groupby("YEAR"):
  year_dir = os.path.join(ERA5_BASE_DIR, str(year))

  # Cari semua file .nc results ekstraksi (mengabaikan file .zip asli)
  nc_files = [
      f
      for f in glob.glob(
          os.path.join(year_dir, "**", "*.nc"), recursive=True
      )
      if not f.endswith(".zip")
  ]

  if not nc_files:
    print(f"Warning: .nc files for year {year} were not found.")
    extracted_df_list.append(group)
    continue

  print(
      f"Processing year {year} ({len(group)} cyclone points) from"
      f" {len(nc_files)} NetCDF files..."
  )

  try:
    # Membuka valuesset NetCDF
    ds = xr.open_mfvaluesset(
        nc_files, combine="by_coords", combine_attrs="override"
    )

    # Automatically detect coordinate variable names in the NetCDF file
    lat_name = next(
        (c for c in ["latitude", "lat", "Y"] if c in ds.coords), None
    )
    lon_name = next(
        (c for c in ["longitude", "lon", "X"] if c in ds.coords), None
    )
    time_name = next(
        (c for c in ["time", "valid_time", "valid_time0"] if c in ds.coords),
        None,
    )

    if not lat_name or not lon_name or not time_name:
      print(
          f"Failed to detect coordinate names for year {year}. Available coords:"
          f" {list(ds.coords)}"
      )
      extracted_df_list.append(group)
      ds.close()
      continue

    # Adjust longitude range if stored in 0° - 360° format to -180° - 180°
    if float(ds[lon_name].max()) > 180:
      ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
      ds = ds.sortby(lon_name)

    var_names = list(ds.values_vars)
    extracted_rows = []

    # Iterate through each tropical cyclone point
    for idx, row in tqdm(
        group.iterrows(), total=len(group), desc=f"Year {year}"
    ):
      row_dict = row.to_dict()
      try:
        # Extract using nearest-neighbor matching (spatial & temporal)
        sub_ds = ds.sel(
            {
                time_name: row["ISO_TIME"],
                lat_name: row["LAT"],
                lon_name: row["LON"],
            },
            method="nearest",
        )

        for var in var_names:
          val = sub_ds[var].values
          row_dict[f"era5_{var}"] = (
              float(val.flatten()[0]) if val.size > 1 else float(val)
          )

      except Exception:
        for var in var_names:
          row_dict[f"era5_{var}"] = None

      extracted_rows.append(row_dict)

    extracted_df_list.append(pd.DataFrame(extracted_rows))
    ds.close()

  except Exception as err:
    print(f"Failed to process NetCDF files for year {year}: {err}")
    extracted_df_list.append(group)

# ==========================================
# 4. MERGE DATA AND SAVE TO EXCEL
# ==========================================
final_df = pd.concat(extracted_df_list, ignore_index=True)
final_df.drop(columns=["YEAR"], inplace=True, errors="ignore")

final_df.to_excel(OUTPUT_PATH, index=False)
print(f"\nProses Selesai! File results ekstraksi disimpan di: {OUTPUT_PATH}")

