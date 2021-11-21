import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sklearn
import seaborn
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import ttest_ind
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import catboost
from sklearn.metrics import classification_report
from config import CBSolverConfig
import re
import numpy as np
from sklearn.cluster import AffinityPropagation
import distance


class Solver:
    def __init__(self, config: CBSolverConfig):
        self.clf = catboost.CatBoostClassifier()
        self.config = config
        self.clf.load(self.config.model_path)

    def upload_df(self, xls_file):
        df = pd.ExcelFile(xls_file, engine="openpyxl")
        md = df.parse(self.config.mon, na_values=["NA"], names=self.config.header)
        for idx, val in enumerate(md["Км"]):
            if not isinstance(val, float):
                md.drop(index=idx, inplace=True)
        md["file"] = xls_file
        self.full_df = md
        return True

    def preproc_df(self):
        self.full_df["point"] = pd.to_numeric(self.full_df["Км"]) * 1000 + pd.to_numeric(self.full_df["М"])
        regex = re.compile("[^а-яА-я]")
        self.full_df["Дефект"] = self.full_df["Дефект"].apply(lambda x: regex.sub("", str(x)))
        self.full_df["Дефект"] = self.full_df["Дефект"].str.replace("Кустоваянегодность", "")
        self.full_df["Дефект"] = self.full_df["Дефект"].str.lower()
        words = self.full_df["Дефект"].unique()
        lev_similarity = -1 * np.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])

        affprop = AffinityPropagation(affinity="precomputed", damping=0.6)
        affprop.fit(lev_similarity)
        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            self.full_df["Дефект"][self.full_df["Дефект"].isin(cluster)] = exemplar
        self.full_df["Статус"] = self.full_df["Статус"].str.replace("Не Подтвержден", "нп")
        self.full_df["Статус"] = self.full_df["Статус"].str.replace("Не подтвержден", "нп")
        self.full_df["Статус"] = self.full_df["Статус"].str.replace("не подтвержден", "нп")
        self.full_df["Статус"] = self.full_df["Статус"].str.replace("Удален Автокорректировкой", "уа")
        self.full_df["Статус"] = self.full_df["Статус"].str.replace("Удален автокорректировкой", "уа")
        self.full_df["Статус"] = self.full_df["Статус"].str.replace("уа", "нп")
        target_df = pd.DataFrame()
        target_df["obj"] = self.config.le_obj.fit_transform((self.full_df["Объект"]))
        target_df["side"] = self.config.le_side.transform((self.full_df["Сторона"]))
        target_df["file"] = self.config.le_file.transform(self.full_df["file"])
        target_df["type"] = self.config.le_type.transform(self.full_df["Тип"])
        target_df["def"] = self.config.le_def.transform(self.full_df["Дефект"])
        return target_df

    def predict(self, target_df: pd.DataFrame):
        target_df["status"] = self.clf.predict(target_df)
        return target_df
