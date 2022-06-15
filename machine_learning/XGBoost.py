import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from errors import exceptions as ex


class StudyXGBoost:
    def __init__(self):
        super().__init__()

    def help(self):
        print("Explain Method:\n"
              "\texplain_boosting".ljust(30), "\tBoosting 기법에 대해 설명합니다.\n" +
              "\texplain_xg_boost", "\tXGBoost 라이브러리에 대해 설명합니다.\n" +
              "\texplain_advantage", "\tXGBoost 라이브러리의 장점에 대해 설명합니다.\n"
              )

        print("Train Method:\n"
              "\ttrain".ljust(30), "\t데이터를 넣고 실제 학습을 돌립니다.\n"
              )

    def explain_boosting(self):
        print("Boosting 이란?".center(100, "-") +
              "\n여러개의 약한 결정 트리를 조합해 사용하는 Ensemble 기법 중 한개입니다.\n약한 예측 모형의 학습 에러의 "
              "가중치를 두고, 순차적으로 다음 학습 모델에 반영 후 강한 예측모형을 만드는 기법입니다.\n")

    def explain_xg_boost(self):
        print("XGBoost(Extreme Gradient Boosting)이란?".center(100, "-") +
              "\nBoosting 기법을 이용하여 구현한 알고리즘은 대표적으로 Gradient Boost가 있습니다.\n"
              "XGBoost는 이 Gradient Boost의 병렬 학습이 지원되도록 후현한 라이브러리입니다.\n"
              "Regression(회귀), Classification(분류) 문제를 모두 지원하며, 성능과 자원 효율이 좋은편에 속하는 알고리즘입니다.\n"
              )

    def explain_advantage(self):
        print("XGBoost의 장점?".center(100, "-") +
              "\n1.GBM 대비 빠른 수행시간을 가집니다.\n"
              "2.병렬 처리로 학습, 분류 속도가 빠릅니다.\n"
              "3.과적합을 규제합니다.\n"
              "4.분류와 회귀영역에서 뛰어난 예측 성능을 발휘합니다.\n"
              "5.CART (Classification and Regression tree) 앙상블 모델을 사용합니다.\n"
              "6.조기종료 (Early Stopping) 기능이 있습니다.\n"
              "7.다양한 옵션을 제공하여 Customizing이 용이합니다.\n"
              "8.결측치를 내부적으로 처리해줍니다.\n"
              )

    def explain_train_method(self):
        print("학습 방법에 관하여".center(100, "-") +
              "\n모델을 학습시키는 방법은 크게 회귀와 분류로 나뉘어집니다.\n"
              "우선 회귀는 보통 0과 1사이의 연속된 값을 예측하는 경우에 사용합니다.\n"
              "이와 반대로 분류 같은 경우는 특정 범주 내의 값중에서 결과를 예측하는 경우 사용합니다.\n"
              "만약 StudyXGBoost에서 학습 방법을 자동으로 설정할 시 유일값(유니크) 값의 개수에 따라서 \n"
              "10개 이하는 분류 모델, 10개 초과는 회귀모델로 학습이 진행됩니다. \n"
              )

    def get_hyper_param_structure(self):
        pass

    def set_train_method(self, train_method):
        train_method_info = {'회귀': XGBRegressor, '분류': XGBClassifier, '자동 설정': 'auto'}

        if train_method not in train_method_info.keys():
            raise ex.NotFoundTrainMethod(train_method)

    def train(self, df: pd.DataFrame, label: str):
        if label not in df.columns:
            raise ex.NotFoundLabelColumnEx(label, df.columns)
        print(df)


if __name__ == '__main__':
    # XGBoost().explain_boosting()
    # XGBoost().explain_xg_boost()
    # XGBoost().explain_advantage()

    xgboost = StudyXGBoost()
    xgboost.explain_train_method()
    xgboost.set_train_method("temp")
    # xgboost.train(df, label)
    # df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    # XGBoost().train(df, 'C')
