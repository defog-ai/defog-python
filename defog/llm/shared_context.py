"""Shared context store for cross-agent communication and knowledge sharing."""

import asyncio
import json
import logging
import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiofiles
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# Removed SecurityError - using standard exceptions instead


from .config.enums import ArtifactType


def _serialize_for_json(obj: Any) -> Any:
    """Recursively serialize objects for JSON, handling datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Handle objects with __dict__ (like dataclass instances)
        return _serialize_for_json(obj.__dict__)
    else:
        return obj


@dataclass
class Artifact:
    """Represents a stored artifact with metadata."""

    key: str
    content: Any
    artifact_type: ArtifactType
    agent_id: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    version: int = 1
    parent_key: Optional[str] = None  # For tracking artifact lineage


class SharedContextStore:
    """
    Filesystem-based shared context store for agent communication.

    Provides a centralized storage system where agents can:
    - Write artifacts that other agents can read
    - Track artifact lineage and versions
    - Store different types of content (plans, results, explorations)
    - Maintain agent namespacing while allowing cross-agent access
    """

    def __init__(self, base_path: str = ".agent_workspace", max_file_size_mb: int = 10):
        # Basic configuration
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Create base path
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(exist_ok=True)

        # Create subdirectories for organization
        self.artifacts_dir = self.base_path / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self.metadata_dir = self.base_path / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # In-memory cache for frequently accessed artifacts
        self._cache: Dict[str, Artifact] = {}
        self._cache_size_limit = 100

        logger.info(f"SharedContextStore initialized at {self.base_path}")

    # Removed security validation methods - simplified implementation

    def _validate_content_size(self, content: Any) -> None:
        """Validate content size."""
        # Estimate content size (rough approximation)
        if isinstance(content, str):
            size = len(content.encode("utf-8"))
        elif isinstance(content, (dict, list)):
            size = len(json.dumps(content, default=str).encode("utf-8"))
        else:
            size = len(str(content).encode("utf-8"))

        if size > self.max_file_size_bytes:
            raise ValueError(
                f"Content size ({size} bytes) exceeds maximum allowed size ({self.max_file_size_bytes} bytes)"
            )

    def _get_artifact_path(self, key: str) -> Path:
        """Get the file path for an artifact."""
        # Use first 2 chars of key hash for directory sharding
        key_hash = hashlib.md5(key.encode()).hexdigest()
        shard = key_hash[:2]
        shard_dir = self.artifacts_dir / shard
        shard_dir.mkdir(exist_ok=True)

        # Use the full hash as filename to avoid filesystem issues
        return shard_dir / f"{key_hash}.json"

    def _get_metadata_path(self, key: str) -> Path:
        """Get the metadata file path for an artifact."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.metadata_dir / f"{key_hash}.meta.json"

    async def write_artifact(
        self,
        agent_id: str,
        key: str,
        content: Any,
        artifact_type: ArtifactType,
        metadata: Optional[Dict[str, Any]] = None,
        parent_key: Optional[str] = None,
    ) -> Artifact:
        """
        Write an artifact to the shared store.

        Args:
            agent_id: ID of the agent writing the artifact
            key: Unique key for the artifact
            content: The content to store
            artifact_type: Type of artifact being stored
            metadata: Optional metadata about the artifact
            parent_key: Optional key of parent artifact (for lineage tracking)

        Returns:
            The created Artifact object
        """
        # Basic validation
        self._validate_content_size(content)

        async with self._lock:
            # Check if artifact already exists (for versioning)
            existing = await self._read_artifact_internal(key)
            version = 1
            if existing:
                version = existing.version + 1
                # Archive the old version
                await self._archive_artifact(existing)

            # Create artifact object
            artifact = Artifact(
                key=key,
                content=content,
                artifact_type=artifact_type,
                agent_id=agent_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata=metadata,
                version=version,
                parent_key=parent_key,
            )

            # Write to filesystem
            artifact_path = self._get_artifact_path(key)
            artifact_dict = asdict(artifact)
            # Convert datetime to ISO format
            artifact_dict["created_at"] = artifact.created_at.isoformat()
            artifact_dict["updated_at"] = artifact.updated_at.isoformat()
            artifact_dict["artifact_type"] = artifact.artifact_type.value
            # Serialize content to handle datetime objects recursively
            artifact_dict["content"] = _serialize_for_json(artifact_dict["content"])

            try:
                # Ensure parent directory exists
                artifact_path.parent.mkdir(parents=True, exist_ok=True)

                # Write file
                async with aiofiles.open(artifact_path, "w", encoding="utf-8") as f:
                    await f.write(
                        json.dumps(artifact_dict, indent=2, ensure_ascii=False)
                    )

            except Exception as e:
                logger.error(f"Failed to write artifact '{key}': {e}")
                raise ValueError(f"Failed to write artifact: {e}")

            # Update cache
            self._cache[key] = artifact
            self._manage_cache_size()

            # Log the write
            logger.debug(f"Agent {agent_id} wrote artifact '{key}' (v{version})")

            return artifact

    async def read_artifact(
        self, key: str, agent_id: Optional[str] = None
    ) -> Optional[Artifact]:
        """
        Read an artifact from the shared store.

        Args:
            key: The key of the artifact to read
            agent_id: Optional agent ID for logging purposes

        Returns:
            The Artifact object if found, None otherwise
        """
        # Check cache first
        if key in self._cache:
            if agent_id:
                logger.debug(f"Agent {agent_id} read artifact '{key}' from cache")
            return self._cache[key]

        # Read from filesystem
        artifact = await self._read_artifact_internal(key)

        if artifact:
            # Update cache
            self._cache[key] = artifact
            self._manage_cache_size()

            if agent_id:
                logger.debug(f"Agent {agent_id} read artifact '{key}'")

        return artifact

    async def _read_artifact_internal(self, key: str) -> Optional[Artifact]:
        """Internal method to read artifact from filesystem."""
        artifact_path = self._get_artifact_path(key)

        if not artifact_path.exists():
            return None

        try:
            async with aiofiles.open(artifact_path, "r") as f:
                data = json.loads(await f.read())

            # Convert back to Artifact object
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            data["artifact_type"] = ArtifactType(data["artifact_type"])

            return Artifact(**data)
        except Exception as e:
            logger.error(f"Error reading artifact {key}: {e}")
            return None

    async def list_artifacts(
        self,
        pattern: Optional[str] = None,
        agent_id: Optional[str] = None,
        artifact_type: Optional[ArtifactType] = None,
        since: Optional[datetime] = None,
    ) -> List[Artifact]:
        """
        List artifacts matching the given criteria.

        Args:
            pattern: Optional glob pattern for key matching
            agent_id: Filter by agent ID
            artifact_type: Filter by artifact type
            since: Only return artifacts created after this time

        Returns:
            List of matching artifacts
        """
        artifacts = []

        # Iterate through all shards
        for shard_dir in self.artifacts_dir.iterdir():
            if not shard_dir.is_dir():
                continue

            for artifact_file in shard_dir.glob("*.json"):
                # Read the artifact to get the original key
                try:
                    async with aiofiles.open(artifact_file, "r") as f:
                        data = json.loads(await f.read())

                    key = data.get("key", "")

                    # Apply pattern filter using fnmatch for glob-style patterns
                    if pattern and not fnmatch.fnmatch(key, pattern):
                        continue

                    # Convert back to Artifact object
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                    data["updated_at"] = datetime.fromisoformat(data["updated_at"])
                    data["artifact_type"] = ArtifactType(data["artifact_type"])

                    artifact = Artifact(**data)

                except Exception as e:
                    logger.warning(f"Error reading artifact file {artifact_file}: {e}")
                    continue

                # Apply filters
                if agent_id and artifact.agent_id != agent_id:
                    continue

                if artifact_type and artifact.artifact_type != artifact_type:
                    continue

                if since and artifact.created_at < since:
                    continue

                artifacts.append(artifact)

        # Sort by creation time (newest first)
        artifacts.sort(key=lambda a: a.created_at, reverse=True)

        return artifacts

    async def get_artifact_lineage(self, key: str) -> List[Artifact]:
        """
        Get the lineage of an artifact (all ancestors).

        Args:
            key: The artifact key

        Returns:
            List of artifacts in lineage order (oldest to newest)
        """
        lineage = []
        current_key = key
        seen_keys = set()  # Prevent circular references

        while current_key and current_key not in seen_keys:
            artifact = await self.read_artifact(current_key)
            if not artifact:
                break

            lineage.append(artifact)
            seen_keys.add(current_key)
            current_key = artifact.parent_key

        # Reverse to get oldest first
        lineage.reverse()

        return lineage

    async def delete_artifact(self, key: str, agent_id: str) -> bool:
        """
        Delete an artifact from the store.

        Args:
            key: The artifact key
            agent_id: The agent requesting deletion

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            artifact_path = self._get_artifact_path(key)

            if not artifact_path.exists():
                return False

            # Archive before deletion
            artifact = await self._read_artifact_internal(key)
            if artifact:
                await self._archive_artifact(artifact)

            # Delete the file
            artifact_path.unlink()

            # Remove from cache
            if key in self._cache:
                del self._cache[key]

            logger.info(f"Agent {agent_id} deleted artifact '{key}'")

            return True

    async def _archive_artifact(self, artifact: Artifact):
        """Archive an artifact before overwriting or deletion."""
        archive_dir = self.base_path / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Create archive filename with timestamp and version
        timestamp = artifact.updated_at.strftime("%Y%m%d_%H%M%S")
        archive_name = f"{artifact.key}_v{artifact.version}_{timestamp}.json"
        archive_path = archive_dir / archive_name

        # Write to archive
        artifact_dict = asdict(artifact)
        artifact_dict["created_at"] = artifact.created_at.isoformat()
        artifact_dict["updated_at"] = artifact.updated_at.isoformat()
        artifact_dict["artifact_type"] = artifact.artifact_type.value

        async with aiofiles.open(archive_path, "w") as f:
            await f.write(json.dumps(artifact_dict, indent=2))

    def _manage_cache_size(self):
        """Manage cache size by removing oldest entries if needed."""
        if len(self._cache) > self._cache_size_limit:
            # Remove oldest entries (by access order)
            items_to_remove = len(self._cache) - self._cache_size_limit
            for key in list(self._cache.keys())[:items_to_remove]:
                del self._cache[key]

    async def clear_all(self, confirm: bool = False):
        """
        Clear all artifacts from the store.

        Args:
            confirm: Must be True to actually clear
        """
        if not confirm:
            logger.warning("clear_all called without confirmation")
            return

        async with self._lock:
            # Archive everything first
            artifacts = await self.list_artifacts()
            for artifact in artifacts:
                await self._archive_artifact(artifact)

            # Clear artifacts directory
            for shard_dir in self.artifacts_dir.iterdir():
                if shard_dir.is_dir():
                    for file in shard_dir.iterdir():
                        file.unlink()

            # Clear cache
            self._cache.clear()

            logger.info("SharedContextStore cleared")

    async def get_agent_artifacts(self, agent_id: str) -> List[Artifact]:
        """Get all artifacts created by a specific agent."""
        return await self.list_artifacts(agent_id=agent_id)

    async def get_recent_artifacts(
        self, limit: int = 10, artifact_type: Optional[ArtifactType] = None
    ) -> List[Artifact]:
        """Get the most recent artifacts."""
        artifacts = await self.list_artifacts(artifact_type=artifact_type)
        return artifacts[:limit]
