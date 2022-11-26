from sklearn.datasets import *
import pandas as pd


class ClassificationDataset:
    def __init__(self):
        self.dataset = None
        self.classification_dataset_list = ['load_iris', 'load_digits', 'load_wine', 'load_breast_cancer',
                                            'fetch_olivetti_faces', 'fetch_20newsgroups',
                                            'fetch_20newsgroups_vectorized', 'fetch_lfw_people', 'fetch_lfw_pairs',
                                            'fetch_covtype', 'fetch_rcv1', 'fetch_kddcup99', 'fetch_california_housing']
        self.dataset_object = {
            'load_iris': load_iris,
            'load_digits': load_digits,
            'load_wine': load_wine,
            'load_breast_cancer': load_breast_cancer,
            'fetch_olivetti_faces': fetch_olivetti_faces,
            'fetch_20newsgroups': fetch_20newsgroups,
            'fetch_20newsgroups_vectorized': fetch_20newsgroups_vectorized,
            'fetch_lfw_people': fetch_lfw_people,
            'fetch_lfw_pairs': fetch_lfw_pairs,
            'fetch_covtype': fetch_covtype,
            'fetch_rcv1': fetch_rcv1,
            'fetch_kddcup99': fetch_kddcup99,
            'fetch_california_housing': fetch_california_housing,
        }

    def load_data(self, data_name: str) -> None:
        self.dataset = self.dataset_object[data_name]()

        if self.dataset is None:
            raise ValueError(f'data_name must be in {self.classification_dataset_list}')

    def show_available_dataset(self, print_flag: bool = True) -> list[str]:
        if print_flag:
            print(self.classification_dataset_list)
        return self.classification_dataset_list

    def get_feed_data(self, return_type: str = 'data_frame'):
        if self.dataset is None:
            raise ValueError('You must first import the data using "load_data" method. ')

        if return_type == 'data_frame':
            return pd.DataFrame(self.dataset['data'], columns=self.dataset['feature_names'])
        elif return_type == 'records':
            return pd.DataFrame(self.dataset['data'], columns=self.dataset['feature_names']).to_dict('records')
        elif return_type == 'dict':
            return pd.DataFrame(self.dataset['data'], columns=self.dataset['feature_names']).to_dict()
        elif return_type == 'array':
            return self.dataset['data']
        else:
            raise ValueError(
                f'data_name must be in ["data_frame", "records", "dict", "array"].\nBut you entered "{return_type}"')


if __name__ == '__main__':
    test_cls = ClassificationDataset()
    test_cls.load_data('load_iris')
    temp = test_cls.get_feed_data('array2')
