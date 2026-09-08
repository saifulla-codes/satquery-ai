"""RSVQA (Remote Sensing Visual Question Answering) and CDVQA Adapters."""

import os
import json
from typing import Dict, Any, List, Optional
from ..base_adapter import BaseDatasetAdapter


class RSVQAAdapter(BaseDatasetAdapter):
    """Adapter for RSVQA benchmark (Low Resolution Sentinel-2 and High Resolution Aerial)."""

    def __init__(self, root_dir: Optional[str] = None):
        super().__init__(root_dir)
        self._qa_index: Optional[Dict[str, List[Dict[str, str]]]] = None

    @property
    def name(self) -> str:
        return "RSVQA"

    @property
    def dataset_type(self) -> str:
        return "Remote Sensing Visual Question Answering"

    def _default_env_root(self) -> Optional[str]:
        return os.getenv("DATASET_ROOT_RSVQA")

    def _load_index(self):
        if self._qa_index is None:
            self._qa_index = {}
            if self.is_available and self.root_dir:
                questions_file = os.path.join(self.root_dir, "questions.json")
                answers_file = os.path.join(self.root_dir, "answers.json")
                if os.path.exists(questions_file) and os.path.exists(answers_file):
                    with open(questions_file, "r") as q_f, open(answers_file, "r") as a_f:
                        q_data = json.load(q_f).get("questions", [])
                        a_data = json.load(a_f).get("answers", [])
                        ans_map = {a["id"]: a.get("answer", "") for a in a_data}
                        for q in q_data:
                            img_id = q.get("image_id", "")
                            if img_id not in self._qa_index:
                                self._qa_index[img_id] = []
                            self._qa_index[img_id].append({
                                "type": q.get("type", "general"),
                                "question": q.get("question", ""),
                                "answer": ans_map.get(q.get("id"), ""),
                            })

    def list_samples(self) -> List[str]:
        if not self.is_available:
            return []
        self._load_index()
        return list(self._qa_index.keys()) if self._qa_index else []

    def load_sample(self, sample_id: str) -> Dict[str, Any]:
        if not self.is_available:
            raise FileNotFoundError(
                "RSVQA dataset directory not mounted. Mount benchmark and configure DATASET_ROOT_RSVQA."
            )
        self._load_index()
        if self._qa_index and sample_id in self._qa_index:
            return {
                "image_id": sample_id,
                "questions": self._qa_index[sample_id],
            }
        raise KeyError(f"Sample ID '{sample_id}' not found in RSVQA index.")

    def load_questions_for_image(self, image_id: str) -> List[Dict[str, str]]:
        if not self.is_available:
            return []
        self._load_index()
        return self._qa_index.get(image_id, []) if self._qa_index else []


class CDVQAAdapter(BaseDatasetAdapter):
    """Adapter for CDVQA (Change Detection Visual Question Answering) benchmark."""

    def __init__(self, root_dir: Optional[str] = None):
        super().__init__(root_dir)
        self._pairs_index: Optional[Dict[str, List[Dict[str, str]]]] = None

    @property
    def name(self) -> str:
        return "CDVQA"

    @property
    def dataset_type(self) -> str:
        return "Bitemporal Change Detection Visual Question Answering"

    def _default_env_root(self) -> Optional[str]:
        return os.getenv("DATASET_ROOT_CDVQA")

    def _load_index(self):
        if self._pairs_index is None:
            self._pairs_index = {}
            if self.is_available and self.root_dir:
                qa_file = os.path.join(self.root_dir, "change_qa.json")
                if os.path.exists(qa_file):
                    with open(qa_file, "r") as f:
                        data = json.load(f)
                        self._pairs_index = data.get("pairs", {})

    def list_samples(self) -> List[str]:
        if not self.is_available:
            return []
        self._load_index()
        return list(self._pairs_index.keys()) if self._pairs_index else []

    def load_sample(self, sample_id: str) -> Dict[str, Any]:
        if not self.is_available:
            raise FileNotFoundError(
                "CDVQA dataset directory not mounted. Mount benchmark and configure DATASET_ROOT_CDVQA."
            )
        self._load_index()
        if self._pairs_index and sample_id in self._pairs_index:
            return {
                "pair_id": sample_id,
                "qa_pairs": self._pairs_index[sample_id],
            }
        raise KeyError(f"Pair ID '{sample_id}' not found in CDVQA index.")

    def load_change_qa(self, pair_id: str) -> List[Dict[str, str]]:
        if not self.is_available:
            return []
        self._load_index()
        return self._pairs_index.get(pair_id, []) if self._pairs_index else []
