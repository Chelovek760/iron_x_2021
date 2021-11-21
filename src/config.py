from dataclasses import dataclass

import numpy as np
from sklearn import preprocessing


@dataclass
class SegmentationConfig:
    threshold = 0.8
    model_weights = ""


@dataclass
class CBSolverConfig:
    model_path = ""
    mon = "Ноябрь"
    header = ["№", "Объект", "П-Н", "Км", "М", "Сторона", "Параметр", "Тип", "Дефект", "Огр.скорости (км/ч)", "Статус"]
    le_def = preprocessing.LabelEncoder()
    le_status = preprocessing.LabelEncoder()
    le_file = preprocessing.LabelEncoder()
    le_side = preprocessing.LabelEncoder()
    le_type = preprocessing.LabelEncoder()
    le_obj = preprocessing.LabelEncoder()
    le_def.classes_ = np.load("")
    le_status.classes_ = np.load("")
    le_file.classes_ = np.load("")
    le_side.classes_ = np.load("")
    le_type.classes_ = np.load("")
    le_obj.classes_ = np.load("")
