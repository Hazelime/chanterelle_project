# Predicting Chanterelle Observations with Place Names
This repository contains code related to a project in the course Machine Learning for Statistical NLP: Advanced (LT2326) at Gothenburg University. Please read final_report.pdf for more context.

# Prerequisites
In order to use the full repository as intended, the following preliminary steps are necessary:
1. Download the environmental 10 x 10 m raster of Sweden (4.80 GB) from https://geodatakatalogen.naturvardsverket.se/geonetwork/srv/swe/catalog.search#/metadata/8853721d-a466-4c01-afcc-9eae57b17b39  and put it in ./data/NMD2023oct2018_basskikt_v0_3.
2. Download daily 4 x 4 km Swedish precipitation and temperature data for 1994-2025 (8.68 GB) from https://www.smhi.se/data/nederbord-och-fuktighet/nederbord/griddad-nederbord--och-temperaturdata and put it in ./data/weather/precipitation and ./data/weather/temperature, respectively.
3. Download Swedish place name data (152 MB) from https://www.lantmateriet.se/sv/geodata/vara-produkter/produktlista/ortnamn-nedladdning-vektor and put it in ./data/placenames/placenames_raw.gpkg.
4. Download the pre-trained Swedish fastText embedding model (7.06 GB) from https://fasttext.cc/docs/en/crawl-vectors.html and put it in ./data/placenames/fasttext/cc.sv.300.bin.
5. Make sure these non-standard Python libraries are installed: numpy, pandas, geopandas, fasttext, rasterio, pyarrow, pyproj, sklearn, xarray, scipy, matplotlib, joblib, and shapely.

# Script Order
To replicate the experiment in the study, the following usage/build order is recommended:

## Building Basic Data
1. **build_grid.py** – Builds the 10 x 10 km master grid using ./data/NMD2023och2018_basskikt_v0_3/NMD2023och2018bas_v0_3.tif, calculates forest coverage for each cell, and saves the data as ./data/sweden_master_grid.csv
2. **smhi_to_grid_mapping.py** – Creates a mapping from the SMHI 4 x 4 km grid to the 10 x 10 km master grid and saves it in ./data/weather/smhi_to_master_grid.parquet.
3. **build_weekly_weather.py** – Calculates average temperature and total precipitation for the preceding weeks for all weeks during 1995-2025 using data from ./data/weather, aligns the result with the master grid, and saves it as ./data/sweden_weekly_weather.parquet.
4. **build_observation_data.py** – Downloads chanterelle observations data from https://api.gbif.org/v1/occurrence/search, filters it to clean observations during 1995-2025, aligns it with the master grid, and saves it to ./data/observations.parquet.
5. **build_placename_data.py** – Extracts placename data from ./data/placenames/placenames_raw.gpkg, aligns it with the master grid, and saves it in ./data/placenames/cell_placenames.parquet.
6. **build_cell_week_data.p**y – Aggregates cell ID, forest coverage, weather, observation, placename, and week data in the joint, compressed ./data/processed/cell_week_dataset.parquet.

## Building Complex Data
7. **tfidf_features.py** – Provides high-level functions for fitting a TF-IDF vectorizer and building TF-IDF features.
8. **build_fasttext_embeddings.py** – Generates cell-level FastText placename embeddings based on ./data/placenames/fasttext/cc.sv.300.bin and ./data/placenames/cell_placenames.parquet and then saves them as ./data/placenames/cell_fasttext_embeddings.parquet.

## Training Models
9. **cell_feature_model.py** – Provides a custom-made logistic-regression model class that handles recurring data more efficiently than Scikit-Learn's LogisticRegression.
10. **test_cell_feature_model.py** – Tests whether the custom-made logistic-regression model produces similar results to Scikit-Learn's LogisticRegression model.
11. **build_splits.py** – Splits the data in ./data/processed/cell_week_dataset.parquet into temporal and spatial training, validation, and testing splits and saves the mappings in ./data/splits.
12. **train_model.py** – Trains and evaluates models based on user input. The available models are seasonality, environmental, placename_count, placename_count_log, tfidf, and fasttext. The available validation schemes are temporal, spatial, and spatiotemporal. Results are saved in the ./results folder and select models are saved in ./results/models.

## Evaluation
10. **compare_models.py** – Compares statistics from tested models in the ./results folder and plots relevant diagrams. Comparison statistics are saved in the ./results folder and the diagrams are saved in ./results/figures.
11. **inspect_tfidf_coefficients.py** – Calculates top positive and negative TF-IDF features for all cells and saves the results in ./results/tfidf_analysis. Some handpicked patterns (e.g. mad, kull, myr, and jávr) are also analyzed more closely.
12. **plot_temporal_predictions.py** – Plots various heatmaps based on the input and results data and saves them in ./results/maps.
