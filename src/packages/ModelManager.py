import sklearn
from sklearn.metrics import accuracy_score, log_loss, roc_curve
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

VALID_MODELS = ["svm", "logreg", "knn", "decision_tree", "random_forest"]


class ModelManager:
    def __init__(self, model_name, data, feature_cols, merge_train_dev: bool = False):
        self.model_name = model_name
        self.data = data
        self._splitDataframe(merge_train_dev=merge_train_dev)
        self.init_model()
        self.feature_cols = feature_cols

    #model initialization
    def init_model(self, params=None):
        assert (
            self.model_name.lower() in VALID_MODELS
        ), f"{self.model_name} is not valid. Valid models include {VALID_MODELS}"

        if self.model_name == "svm":
            if params is None:
                self.model = SVC()
            else:
                self.model = SVC(**params)
        elif self.model_name == "logreg":
            if params is None:
                self.model = LogisticRegression()
            else:
                self.model = LogisticRegression(**params)
        elif self.model_name == "knn":
            if params is None:
                self.model = KNeighborsClassifier()
            else:
                self.model = KNeighborsClassifier(**params)
        elif self.model_name == "decision_tree":
            if params is None:
                self.model = DecisionTreeClassifier()
            else:
                self.model = DecisionTreeClassifier(**params)
        elif self.model_name == "random_forest":
            if params is None:
                self.model = RandomForestClassifier(random_state=12)
            else:
                self.model = RandomForestClassifier(**params)

    def _splitDataframe(self, merge_train_dev: bool):

        if merge_train_dev:
            self.train = self.data[
                (self.data.type == "train") | (self.data.type == "dev")
            ]
            self.dev = None
        else:
            self.train = self.data[(self.data.type == "train")]
            self.dev = self.data[(self.data.type == "dev")]

        self.test = self.data[(self.data.type == "test")]

    def trainModel(self, label_col: str):
        # Train the model using the training data
        self.y_train = self.train[label_col]
        self.X_train = self.train[self.feature_cols].copy()

        self.X_train.to_csv("/home/ubuntu/features.csv", index=False)

        self.model.fit(self.X_train, self.y_train)

    def predict(self, label_col: str):
        # Make predictions on the test data
        self.y_test = self.test[label_col]
        self.X_test = self.test[self.feature_cols].copy()

        self.y_pred = self.model.predict(self.X_test)

        # Calculate accuracy and log loss
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

        self.class_accuracy = {}
        cls_y_test = self.y_test.copy()
        cls_y_test = cls_y_test.reset_index(drop=True)
        for cls in range(len(set(self.y_test))):
            cls_name = self.data.loc[
                self.data[label_col] == cls, "architecture"
            ].unique()[0]

            cls_idx = np.where(self.y_test == cls)[0]
            cls_test = cls_y_test[cls_idx]
            cls_pred = self.y_pred[cls_idx]
            self.class_accuracy[cls_name] = accuracy_score(cls_test, cls_pred)

        self.eer_score, self.eer_threshold = None, None

        if self.model_name not in ["svm"]:
            self.y_prob = self.model.predict_proba(self.X_test)
            self.log_loss_value = log_loss(self.y_test, self.y_prob)

            #calculate eer score
            if "multi" not in label_col:
                self.eer_score, self.eer_threshold = self.calculate_eer()

            return (
                self.accuracy,
                self.log_loss_value,
                self.eer_score,
                self.eer_threshold,
            )

        self.log_loss_value = None
        return self.accuracy, self.log_loss_value, self.eer_score, self.eer_threshold

    #train and predict using model
    def trainPredict(self, label_col: str):
        self.trainModel(label_col=label_col)
        acc, log_loss, eer_score, eer_threshold = self.predict(label_col=label_col)
        return acc, log_loss, eer_score, eer_threshold

    def plotRocCurve(self):
        # Create a ROC curve plot
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob[:, 1])
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    def plotProbaDistribution(self):
        # Create a histogram of test set probability scores
        plt.hist(self.y_prob)
        plt.xlabel("Probability Score")
        plt.ylabel("Frequency")
        plt.title("Test Set Probability Score Distribution")
        plt.show()

    def calculate_eer(self):
        # Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob[:, 1], pos_label=1)

        # Interpolate the FPR and TPR values
        interpolated = interp1d(fpr, tpr)

        # Find the point where FAR and FRR are equal (EER)
        eer = brentq(lambda x: 1.0 - x - interpolated(x), 0.0, 1.0)

        optimal_threshold = thresholds[np.nanargmin(np.abs((1.0 - tpr) - fpr))]

        return eer, optimal_threshold
