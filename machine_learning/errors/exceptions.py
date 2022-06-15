from pandas import Index

class ModelException(Exception):
    msg: str
    detail: str
    ex: Exception

    def __init__(
        self,
        *,
        msg: str = None,
        detail: str = None,
        ex: Exception = None,
    ):
        self.msg = msg
        self.detail = detail
        self.ex = ex
        super().__init__(ex)


class NotFoundLabelColumnEx(ModelException):
    def __init__(self, label: str, column_names: Index, ex: Exception = None):
        super().__init__(
            msg=f"{label} 컬럼을 데이터에서 찾을 수 없습니다.",
            detail=f"데이터 컬럼 리스트 : {column_names}",
            ex=Exception(f"\"{label}\" 컬럼을 데이터에서 찾을 수 없습니다."
                         f"컬럼 리스트 : \"{column_names}\"") if ex is None else ex,
        )

class NotFoundTrainMethod(ModelException):
    def __init__(self, train_method: str, ex: Exception = None):
        super().__init__(
            ex=Exception(f"입력하신 '{train_method}'는 올바른 학습방법이 아닙니다.\n"
                         f"['회귀', '분류', '자동 설정'] 중 선택해주세요.\n"
                         f"학습방법에 관하여 자세히 알고 싶으시다면 explain_train_method 함수를 호출해주세요.") if ex is None else ex,
        )
