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
