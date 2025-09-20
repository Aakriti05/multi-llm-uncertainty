from torch.utils.data import Dataset
from typing import List, Dict, Any

class PromptCompletionDataset(Dataset):
    """
    A simple dataset yielding prompt-completion pairs.
    """
    def __init__(self, json_list: List[Dict[str, Any]]):
        self.json_list = json_list

    def __len__(self):
        return len(self.json_list["file_key"])

    def __getitem__(self, idx) -> Dict[str, str]:
        return {
            "prompt": self.json_list["prompt"][idx],
            "completion": self.json_list["completion"][idx],
            "chat_history": self.json_list["chat_history"][idx],
            "file_key": self.json_list["file_key"][idx],
            "completion_idx": self.json_list["completion_idx"][idx]
        }