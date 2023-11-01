# directory to store data, the data has to be csv format.
DATA_PATH = "../data/"

# year and dates of data
YEAR = "2023"
DATES = ["0925", "0926", "0927", "0928", "1009", "1010", "1011", "1012", "1013", "1016", "1017", "1018", "1019", "1020", "1023", "1024", "1025"]

# number of days among which data would be combined.
WINDOW_SIZE = 12

# features to be used, "symbol" is mandatory.
INPUT_FEATURES = [
        "symbol",
        "return_mean",
        "vwap_mean",
        "volatility_mean",
        # "20_min_SMA_mean",
        # "PBW_mean",
        # "return_std",
        # "vwap_std",
        # "volatility_std",
        # "20_min_SMA_std",
        # "PBW_std",
        ]

# z-score threshold, a data point with abs(z-score) > Z_SCORE would be identified as an outlier and filtered.
Z_SCORE = 3 

# hyperparameter of agglomerative clustering, two clustering within a distance smaller than this number would be combined.
DISTANCE_THRESHOLD = 0.5

# for results visualization, True for plotting scaled features, False for plotting original features.
SAVE_SCALED_DATA = True
