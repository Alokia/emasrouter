from dataset.base_dataset import BaseDataset, BaseDataLoader, BaseMetric
from typing import List, Union, Self
import json


class HumanEvalDataset(BaseDataset):
    """Load HumanEval dataset from jsonl file

    The HumanEval dataset has the following format:
        - task_id: str. Unique identifier for the task.
        - prompt: str. Questions for querying LLM. Function definitions and their docstring that require code completion.
        - entry_point: str. The name of the function to be completed.
        - canonical_solution: str. The ground truth code completion for the `prompt`.
        - test: str. The test cases for the function completion.
    """
    def __init__(self, path: str = None, data: List = None):
        super().__init__(data=data)
        if data is None:
            self.data = self.load_jsonl_data(path)

    @staticmethod
    def load_jsonl_data(path: str) -> List:
        with open(path, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data

    @classmethod
    def split(cls, path: str, ratio: Union[List, float] = 0.8, shuffle: bool = True, **kwargs) -> List[Self]:
        data = cls.load_jsonl_data(path)
        split_data = super().split(ratio=ratio, shuffle=shuffle, data=data, **kwargs)
        return [cls(data=i) for i in split_data]


class HumanEvalDataLoader(BaseDataLoader):
    def __init__(self, dataset: HumanEvalDataset, batch_size=1, shuffle=False, drop_last=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class HumanEvalMetric(BaseMetric):
    def __call__(self, *args, **kwargs):
        raise NotImplemented


if __name__ == '__main__':
    from utils.print_utils import format_print_dict

    path = "human_eval/HumanEval.jsonl"
    train_set, val_set = HumanEvalDataset.split(path, [0.8, 0.2], shuffle=False)
    print(len(train_set), len(val_set))
    # print(train_set, val_set)
    # format_print_dict(train_set[0])
    # print(train_set[0]["prompt"])
    # print(train_set[0]["test"])
