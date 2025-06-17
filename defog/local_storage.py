"""
Local storage utilities for Defog to replace API-based storage
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import datetime


class LocalStorage:
    """
    Handles local file storage for metadata, glossary, and golden queries
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            # Default to .defog_local in the user's home directory
            self.storage_dir = Path.home() / ".defog_local"
        else:
            self.storage_dir = Path(storage_dir)

        # Create storage directories
        self.storage_dir.mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
        (self.storage_dir / "glossary").mkdir(exist_ok=True)
        (self.storage_dir / "golden_queries").mkdir(exist_ok=True)

    def _get_project_id(self, api_key: Optional[str] = None, db_type: str = "") -> str:
        """Generate a project ID based on db_type or api_key for backward compatibility"""
        if api_key:
            # Use hash of api_key for backward compatibility
            import hashlib

            return hashlib.md5(api_key.encode()).hexdigest()[:8]
        elif db_type:
            return db_type
        else:
            return "default"

    # Metadata methods
    def save_metadata(
        self, metadata: Dict[str, Any], api_key: Optional[str] = None, db_type: str = ""
    ) -> Dict[str, Any]:
        """Save metadata to local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "metadata" / f"{project_id}.json"

        # Add timestamp
        metadata["last_updated"] = datetime.datetime.now().isoformat()

        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"status": "success", "message": "Metadata saved locally"}

    def get_metadata(
        self, api_key: Optional[str] = None, db_type: str = ""
    ) -> Dict[str, Any]:
        """Retrieve metadata from local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "metadata" / f"{project_id}.json"

        if not file_path.exists():
            return {"metadata": {}}

        with open(file_path, "r") as f:
            metadata = json.load(f)

        return {"metadata": metadata}

    # Glossary methods
    def save_glossary(
        self, glossary: str, api_key: Optional[str] = None, db_type: str = ""
    ) -> Dict[str, Any]:
        """Save glossary to local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "glossary" / f"{project_id}.txt"

        with open(file_path, "w") as f:
            f.write(glossary)

        return {"status": "success", "message": "Glossary saved locally"}

    def get_glossary(self, api_key: Optional[str] = None, db_type: str = "") -> str:
        """Retrieve glossary from local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "glossary" / f"{project_id}.txt"

        if not file_path.exists():
            return ""

        with open(file_path, "r") as f:
            return f.read()

    def delete_glossary(
        self, api_key: Optional[str] = None, db_type: str = ""
    ) -> Dict[str, Any]:
        """Delete glossary from local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "glossary" / f"{project_id}.txt"

        if file_path.exists():
            file_path.unlink()
            return {"status": "success", "message": "Glossary deleted"}
        else:
            return {"status": "success", "message": "No glossary found"}

    # Golden queries methods
    def save_golden_queries(
        self,
        golden_queries: List[Dict[str, Any]],
        api_key: Optional[str] = None,
        db_type: str = "",
    ) -> Dict[str, Any]:
        """Save golden queries to local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "golden_queries" / f"{project_id}.json"

        # Load existing queries if any
        existing_queries = []
        if file_path.exists():
            with open(file_path, "r") as f:
                existing_queries = json.load(f)

        # Merge with new queries (update existing ones based on question)
        existing_map = {q["question"]: q for q in existing_queries}
        for query in golden_queries:
            existing_map[query["question"]] = query

        # Save back
        all_queries = list(existing_map.values())
        with open(file_path, "w") as f:
            json.dump(all_queries, f, indent=2)

        return {
            "status": "success",
            "message": f"Saved {len(golden_queries)} golden queries",
        }

    def get_golden_queries(
        self, api_key: Optional[str] = None, db_type: str = ""
    ) -> List[Dict[str, Any]]:
        """Retrieve golden queries from local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "golden_queries" / f"{project_id}.json"

        if not file_path.exists():
            return []

        with open(file_path, "r") as f:
            return json.load(f)

    def delete_golden_queries(
        self,
        golden_queries: List[str],
        api_key: Optional[str] = None,
        db_type: str = "",
    ) -> Dict[str, Any]:
        """Delete specific golden queries from local storage"""
        project_id = self._get_project_id(api_key, db_type)
        file_path = self.storage_dir / "golden_queries" / f"{project_id}.json"

        if not file_path.exists():
            return {"status": "success", "message": "No golden queries found"}

        with open(file_path, "r") as f:
            existing_queries = json.load(f)

        # Filter out queries to delete
        questions_to_delete = set(golden_queries)
        remaining_queries = [
            q for q in existing_queries if q["question"] not in questions_to_delete
        ]

        # Save back
        with open(file_path, "w") as f:
            json.dump(remaining_queries, f, indent=2)

        deleted_count = len(existing_queries) - len(remaining_queries)
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} golden queries",
        }

    # Schema storage (for removed upload functionality)
    def save_schema(
        self,
        schema_data: str,
        filename: str,
        api_key: Optional[str] = None,
        db_type: str = "",
    ) -> Dict[str, Any]:
        """Save schema data to local storage"""
        project_id = self._get_project_id(api_key, db_type)
        schema_dir = self.storage_dir / "schemas" / project_id
        schema_dir.mkdir(parents=True, exist_ok=True)

        file_path = schema_dir / filename
        with open(file_path, "w") as f:
            f.write(schema_data)

        return {"status": "success", "message": f"Schema saved to {file_path}"}
