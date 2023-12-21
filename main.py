from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class Model:
    # Modify your model, default is a linear regression model with random weights

    def __init__(self):
        self.theta = None
        self.model = None
        self.fe = None

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self.fe = FeatureExtractor()
        X_train = self.fe.extract_X_train(X_train)

        xgb_params = {'objective': 'reg:squarederror',
                  'eval_metric': 'rmse', 
                  'colsample_bytree': 0.5,
                  'max_depth': 5,
                  'learning_rate': 0.01, 
                  'min_child_weight': 10, 
                  'n_estimators': 3000, 
                  'reg_alpha': 3, 
                  'reg_lambda': 1, 
                  'subsample': 0.6}

        self.xgb_model = xgb.XGBRegressor(**xgb_params)
        self.xgb_model.fit(X_train, y_train)
        return None

    def predict(self, X_test: pd.DataFrame) -> np.array:

        X_test = self.fe.extract_X_test(X_test)
        xgb_pred = self.xgb_model.predict(X_test)

        return xgb_pred



from copy import deepcopy

class FeatureExtractor():
    def __init__(self):

        self.AGG_VERIFICATION = True
        self.CONDENSE_LOCATION = True
        self.FIND_PER_UNIT = True
        self.USE_KMEANS = True
        self.n_clusters = 48

    def extract_X_train(self, X_train):
        return self.transform(X_train, mode="train")

    def extract_X_test(self, X_test):
        return self.transform(X_test, mode="test")

    def transform(self, X_raw, mode="train"):

        X = deepcopy(X_raw)
        col_names = list(X.columns)

        if self.AGG_VERIFICATION:
            # Aggregate all verification methods.
            verification_cols = [item for item in col_names if "verification" in item]
            X["total_num_verifications"] = X[verification_cols].sum(axis=1)
            #X.drop(verification_cols, axis=1, inplace=True)


        if self.FIND_PER_UNIT:
            # Obtain per unit value for current features
            per_unit_dict = {"accommodates":["beds", "bedrooms", "bathrooms",
                                            "cleaning_fee", "security_deposit"],
                            "bedrooms":["bathrooms"]}
            offset = 0.0001

            for key in per_unit_dict:
                for item in per_unit_dict[key]:
                    X[item+"_per_"+key] = X[item]/(X[key]+offset)

        if self.CONDENSE_LOCATION:
            # Merge similar location tags.
            locations = ["astoria", "arverne", "new york", "bronx", "manhattan",
                        "queens", "bushwick", "far rockaway", "elmhurst", "flush",
                        "forest hill", "east williamsburg", "fresh meadows", "glendale", "gravesend",
                        "green_point", "harlem", "hell's kitchen", "hollis", "howard beach",
                        "jackson heights", "jamaica", "kensington", "kew gradens", "long island",
                        "ny", "nyc", "maspeth", "middle village", 'oakland graden', "park slope",
                        "corona", "ridgewood", "rego park", "sunnyside", "south ozone park",
                        "springfield gardens", "williamsburg", "brooklyn", "saint albans",
                        "bayside", "oakland gardens", "kew gardens", "briarwood", "woodside", "new york city",
                        "staten island", "woodhaven"]

            repeated_col_dict = {}
            for loc in locations:
                repeated_col_dict[loc] = [item for item in col_names if loc.replace(" ", "") in item.lower().replace(" ", "")]

            repeated_col_dict["new york"] += ['紐約', '纽约', 'New Youk纽约', 'New-York', 'Nueva York', 'NUEVA YORK']
            repeated_col_dict["new york city"] += ["nyc", '纽约市']
            repeated_col_dict["brooklyn"] += ['布鲁克林', 'Brooklin', 'Brookly,', 'Brookyln ', 'Brookyn', 'Brookyn ']
            repeated_col_dict["flush"] += ['纽约法拉盛']

            for loc in locations:
                X[loc+"_condensed"] = X[repeated_col_dict[loc]].sum(axis=1).apply(lambda x: 1 if x>0.5 else 0)

            for loc in locations:
                X = X.drop(columns=repeated_col_dict[loc], errors="ignore")

        if self.USE_KMEANS:
            # Create KMeans features
            from sklearn.cluster import KMeans
            X_coords = X[["latitude", "longitude"]].to_numpy()
            if mode=="train":
                kmeans = KMeans(n_init = 20, n_clusters=self.n_clusters).fit(X_coords)
                self.kmeans = kmeans
            else:
                kmeans = self.kmeans
            X["kmeans_cluster"] = kmeans.predict(X_coords)
            centroids = kmeans.cluster_centers_
            distance_to_centroid = np.linalg.norm(X_coords - centroids[kmeans.predict(X_coords)], axis=1)
            X["distance_to_kmeans_center"] = distance_to_centroid

        return X



if __name__ == '__main__':
    model = Model() # Model class imported from your submission
    print("Initializing Model...")
    
    X = pd.read_csv("data_cleaned_train_comments_X.csv")  # pandas Dataframe
    y = pd.read_csv("data_cleaned_train_y.csv")  # pandas Dataframe
    print("Loading Data...")

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Splitting Data...")

    model.train(X_train, y_train) # train your model on the dataset provided to you
    y_pred = model.predict(X_test) # test your model on the hidden test set (pandas Dataframe)
    mse = mean_squared_error(y_test, y_pred) # compute mean squared error
    print("The MSE is :", mse)
