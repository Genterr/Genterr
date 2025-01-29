from typing import Dict, Any, Optional, List, Union, Iterator, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import logging
import shutil
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
import h5py
import yaml
from tqdm import tqdm
import dill
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor

class DatasetError(Exception):
    """Base exception for dataset-related errors"""
    pass

class DatasetNotFoundError(DatasetError):
    """Raised when dataset cannot be found"""
    pass

class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails"""
    pass

class DatasetVersionError(DatasetError):
    """Raised when dataset version is incompatible"""
    pass

class DatasetFormat(Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    HDF5 = "h5"
    NUMPY = "npy"
    TEXT = "txt"
    CUSTOM = "custom"

@dataclass
class DatasetMetadata:
    """Metadata for dataset versioning and tracking"""
    name: str
    version: str
    format: DatasetFormat
    created_at: datetime
    updated_at: datetime
    num_samples: int
    size_bytes: int
    features: List[str]
    labels: List[str]
    split_ratio: Dict[str, float]
    checksum: str
    preprocessing_steps: List[str]
    source: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None

@dataclass
class DatasetConfig:
    """Configuration for dataset management"""
    base_path: Path
    cache_size: int = 1000
    compression: bool = True
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    max_workers: int = 4
    chunk_size: int = 10000

class DatasetManager:
    """
    Manages dataset operations including loading, preprocessing, and versioning.
    Supports various data formats and provides efficient data handling mechanisms.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger("dataset_manager")
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.cache: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Create necessary directories
        self.config.base_path.mkdir(parents=True, exist_ok=True)
        (self.config.base_path / "raw").mkdir(exist_ok=True)
        (self.config.base_path / "processed").mkdir(exist_ok=True)
        (self.config.base_path / "metadata").mkdir(exist_ok=True)
        
        # Load existing dataset metadata
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata for all existing datasets"""
        metadata_path = self.config.base_path / "metadata"
        for meta_file in metadata_path.glob("*.yaml"):
            with open(meta_file, 'r') as f:
                meta_dict = yaml.safe_load(f)
                meta_dict['created_at'] = datetime.fromisoformat(meta_dict['created_at'])
                meta_dict['updated_at'] = datetime.fromisoformat(meta_dict['updated_at'])
                meta_dict['format'] = DatasetFormat(meta_dict['format'])
                metadata = DatasetMetadata(**meta_dict)
                self.datasets[metadata.name] = metadata

    async def create_dataset(
        self,
        name: str,
        data: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]],
        format: DatasetFormat,
        features: List[str],
        labels: List[str],
        description: Optional[str] = None,
        source: Optional[str] = None,
        license: Optional[str] = None
    ) -> DatasetMetadata:
        """Create a new dataset with metadata"""
        try:
            # Convert data to appropriate format
            if isinstance(data, list):
                data = pd.DataFrame(data)
            
            # Generate checksum
            checksum = self._generate_checksum(data)
            
            # Create metadata
            metadata = DatasetMetadata(
                name=name,
                version="1.0.0",
                format=format,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                num_samples=len(data),
                size_bytes=self._calculate_size(data),
                features=features,
                labels=labels,
                split_ratio={
                    "train": 1 - self.config.validation_split - self.config.test_split,
                    "val": self.config.validation_split,
                    "test": self.config.test_split
                },
                checksum=checksum,
                preprocessing_steps=[],
                source=source,
                description=description,
                license=license
            )
            
            # Save dataset
            await self._save_dataset(data, metadata)
            
            # Save metadata
            self._save_metadata(metadata)
            
            self.datasets[name] = metadata
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset {name}: {str(e)}")
            raise DatasetError(f"Dataset creation failed: {str(e)}")

    def _generate_checksum(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """Generate checksum for data validation"""
        if isinstance(data, pd.DataFrame):
            return hashlib.sha256(pd.util.hash_pandas_object(data).values).hexdigest()
        return hashlib.sha256(data.tobytes()).hexdigest()

    def _calculate_size(self, data: Union[pd.DataFrame, np.ndarray]) -> int:
        """Calculate size of dataset in bytes"""
        return data.memory_usage(deep=True).sum() if isinstance(data, pd.DataFrame) else data.nbytes

    async def _save_dataset(self, data: Union[pd.DataFrame, np.ndarray], metadata: DatasetMetadata) -> None:
        """Save dataset to disk"""
        path = self.config.base_path / "raw" / f"{metadata.name}_v{metadata.version}"
        
        if metadata.format == DatasetFormat.CSV:
            if isinstance(data, pd.DataFrame):
                await self._save_dataframe_chunks(data, path.with_suffix('.csv'))
        elif metadata.format == DatasetFormat.PARQUET:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(path.with_suffix('.parquet'))
        elif metadata.format == DatasetFormat.HDF5:
            with h5py.File(path.with_suffix('.h5'), 'w') as f:
                f.create_dataset('data', data=data, compression='gzip' if self.config.compression else None)
        elif metadata.format == DatasetFormat.NUMPY:
            np.save(path.with_suffix('.npy'), data)
        else:
            raise ValueError(f"Unsupported format: {metadata.format}")

    async def _save_dataframe_chunks(self, df: pd.DataFrame, path: Path) -> None:
        """Save large DataFrame in chunks"""
        chunks = [df[i:i + self.config.chunk_size] for i in range(0, len(df), self.config.chunk_size)]
        
        async with aiofiles.open(path, 'w') as f:
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await f.write(chunk.to_csv(index=False))
                else:
                    await f.write(chunk.to_csv(index=False, header=False))

    def _save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save dataset metadata"""
        meta_dict = {
            k: v.value if isinstance(v, Enum) else v.isoformat() if isinstance(v, datetime) else v
            for k, v in vars(metadata).items()
        }
        
        path = self.config.base_path / "metadata" / f"{metadata.name}.yaml"
        with open(path, 'w') as f:
            yaml.dump(meta_dict, f)

    async def load_dataset(
        self,
        name: str,
        version: Optional[str] = None,
        split: Optional[str] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Load dataset from disk"""
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset {name} not found")
            
        metadata = self.datasets[name]
        version = version or metadata.version
        
        # Check cache
        cache_key = f"{name}_v{version}_{split if split else 'full'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        path = self.config.base_path / "raw" / f"{name}_v{version}"
        
        try:
            if metadata.format == DatasetFormat.CSV:
                data = await self._load_csv_chunks(path.with_suffix('.csv'))
            elif metadata.format == DatasetFormat.PARQUET:
                data = pd.read_parquet(path.with_suffix('.parquet'))
            elif metadata.format == DatasetFormat.HDF5:
                with h5py.File(path.with_suffix('.h5'), 'r') as f:
                    data = f['data'][:]
            elif metadata.format == DatasetFormat.NUMPY:
                data = np.load(path.with_suffix('.npy'))
            else:
                raise ValueError(f"Unsupported format: {metadata.format}")
            
            # Validate checksum
            if self._generate_checksum(data) != metadata.checksum:
                raise DatasetValidationError("Dataset checksum validation failed")
            
            # Apply split if requested
            if split:
                data = self._split_dataset(data, split)
            
            # Update cache
            if len(self.cache) >= self.config.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {name}: {str(e)}")
            raise DatasetError(f"Dataset loading failed: {str(e)}")

    async def _load_csv_chunks(self, path: Path) -> pd.DataFrame:
        """Load large CSV files in chunks"""
        chunks = []
        for chunk in pd.read_csv(path, chunksize=self.config.chunk_size):
            chunks.append(chunk)
        return pd.concat(chunks)

    def _split_dataset(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        split: str
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Split dataset according to specified ratios"""
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}")
            
        np.random.seed(self.config.random_seed)
        
        total_samples = len(data)
        split_points = [
            int(total_samples * self.datasets[data.name].split_ratio['train']),
            int(total_samples * (self.datasets[data.name].split_ratio['train'] + self.datasets[data.name].split_ratio['val']))
        ]
        
        if isinstance(data, pd.DataFrame):
            shuffled = data.sample(frac=1)
        else:
            shuffled = np.random.permutation(data)
        
        if split == 'train':
            return shuffled[:split_points[0]]
        elif split == 'val':
            return shuffled[split_points[0]:split_points[1]]
        else:  # test
            return shuffled[split_points[1]:]

    async def update_dataset(
        self,
        name: str,
        data: Union[pd.DataFrame, np.ndarray],
        preprocessing_steps: Optional[List[str]] = None
    ) -> DatasetMetadata:
        """Update existing dataset with new data or preprocessing"""
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset {name} not found")
            
        metadata = self.datasets[name]
        
        # Update version
        version_parts = metadata.version.split('.')
        metadata.version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
        
        # Update metadata
        metadata.updated_at = datetime.utcnow()
        metadata.num_samples = len(data)
        metadata.size_bytes = self._calculate_size(data)
        metadata.checksum = self._generate_checksum(data)
        if preprocessing_steps:
            metadata.preprocessing_steps.extend(preprocessing_steps)
        
        # Save updated dataset and metadata
        await self._save_dataset(data, metadata)
        self._save_metadata(metadata)
        
        # Clear cache
        self.cache.clear()
        
        return metadata

    async def delete_dataset(self, name: str) -> bool:
        """Delete dataset and its metadata"""
        try:
            if name not in self.datasets:
                raise DatasetNotFoundError(f"Dataset {name} not found")
                
            metadata = self.datasets[name]
            
            # Remove files
            path = self.config.base_path / "raw" / f"{name}_v{metadata.version}"
            path.unlink()
            
            # Remove metadata
            meta_path = self.config.base_path / "metadata" / f"{metadata.name}.yaml"
            meta_path.unlink()
            
            # Clear from memory
            del self.datasets[name]
            self.cache = {k: v for k, v in self.cache.items() if not k.startswith(f"{name}_")}
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete dataset {name}: {str(e)}")
            return False

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset {name} not found")
            
        metadata = self.datasets[name]
        return {
            "name": metadata.name,
            "version": metadata.version,
            "format": metadata.format.value,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "num_samples": metadata.num_samples,
            "size_mb": metadata.size_bytes / (1024 * 1024),
            "features": metadata.features,
            "labels": metadata.labels,
            "split_ratio": metadata.split_ratio,
            "preprocessing_steps": metadata.preprocessing_steps,
            "source": metadata.source,
            "description": metadata.description,
            "license": metadata.license
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets"""
        return [self.get_dataset_info(name) for name in self.datasets]

    def validate_dataset(self, name: str) -> bool:
        """Validate dataset integrity"""
        try:
            if name not in self.datasets:
                raise DatasetNotFoundError(f"Dataset {name} not found")
                
            metadata = self.datasets[name]
            path = self.config.base_path / "raw" / f"{name}_v{metadata.version}"
            
            # Check if file exists
            if not path.with_suffix(f".{metadata.format.value}").exists():
                return False
            
            # Load data and verify checksum
            data = asyncio.run(self.load_dataset(name))
            current_checksum = self._generate_checksum(data)
            
            return current_checksum == metadata.checksum
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            return False

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        return f"DatasetManager(datasets={len(self.datasets)}, cache_size={self.config.cache_size})"