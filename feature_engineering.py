from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Nothing to fit; transformation is deterministic.
        return self

    def transform(self, X):
        # Create a copy so as not to alter the original DataFrame
        X = X.copy()
        
        # Determine the datetime source: use 'date_time' column if exists,
        # otherwise assume that the DataFrame index is a DatetimeIndex.
        if 'date_time' in X.columns:
            # Convert the date_time column to datetime (if not already)
            dt = pd.to_datetime(X['date_time'])
        elif isinstance(X.index, pd.DatetimeIndex):
            dt = X.index
        else:
            raise ValueError("No 'date_time' column and index is not a DatetimeIndex")
        
        # Extract month, day, and hour from the datetime source.
        X['month'] = dt.month
        X['day'] = dt.day
        X['hour'] = dt.hour

        # Create sine and cosine features for month, day, and hour.
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

        # Create a season feature based on month.
        def get_season(month):
            if month in [3, 4, 5]:
                return 'summer'
            elif month in [6, 7, 8, 9]:
                return 'monsoon'
            elif month in [10, 11]:
                return 'post-monsoon'
            else:
                return 'winter'
        X['season'] = X['month'].apply(get_season)

        # Create precipitation flag (1 if precipMM > 0, else 0)
        X['precip_flag'] = (X['precipMM'] > 0).astype(int)
        # Create precipitation amount field (can be transformed later if needed)
        X['precip_amount'] = X['precipMM']
        
        return X
