# DMCR Package Documentation

**Data Models Context Reduction (DMCR)** is a Python package designed for research in using datamodels for in-context learning in multi-tasking settings. This package provides a comprehensive framework for building, training, and evaluating datamodels that can reduce context requirements while maintaining performance.

## Table of Contents

- [Package Overview](#package-overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)

## Package Overview

The DMCR package is structured around a modular architecture that separates concerns into distinct components:

- **Data Loading**: Flexible data loading mechanisms for various data sources
- **Models**: LLM interfaces and datamodel implementations
- **Pipelines**: End-to-end processing workflows
- **Evaluators**: Comprehensive evaluation frameworks
- **Vector Stores**: Document storage and retrieval systems
- **Retrievers**: Information retrieval components
- **Utilities**: Supporting tools and configurations

## Architecture

```
dmcr/
├── dataloaders/          # Data loading abstractions
├── datamodels/           # Core datamodels functionality
│   ├── config.py        # Configuration classes
│   ├── evaluator/       # Datamodel evaluation
│   ├── models/          # Datamodel implementations
│   ├── pipeline/        # Datamodel pipelines
│   └── setter/          # Data setting strategies
├── evaluators/          # Performance evaluation
├── models/              # LLM interfaces
├── pipelines/           # Processing workflows
├── retrievers/          # Information retrieval
├── utils/               # Utility functions
└── vector_stores/       # Document storage
```

## Core Components

### 1. Data Loaders (`dataloaders/`)

Provides abstract interfaces for loading data from various sources.

- **`BaseDataloader`**: Abstract base class for all data loaders
- **`PartialCSVDataLoader`**: Specialized loader for CSV data with partial loading capabilities

```python
from dmcr.dataloaders import BaseDataloader

class CustomDataloader(BaseDataloader):
    def get_documents(self):
        # Implementation specific logic
        pass
```

### 2. Datamodels (`datamodels/`)

The core of the package, containing datamodel implementations and training pipelines.

#### Configuration (`config.py`)
- **`DatamodelConfig`**: Base configuration for datamodels
- **`DatamodelIndexBasedConfig`**: Configuration for index-based datamodels
- **`MemMapConfig`**: Memory mapping configuration
- **`LogConfig`**: Logging configuration for experiments

#### Models (`models/`)
- **`LinearRegressor`**: Basic linear regression datamodel
- **`LASSOLinearRegressor`**: LASSO regularized linear regression
- **`FactoryModels`**: Factory pattern for creating datamodel instances

#### Pipelines (`pipeline/`)
- **`DatamodelPipeline`**: Main pipeline for datamodel training and inference
- **`TrainModelsPipeline`**: Pipeline for training multiple datamodels
- **`PreCollectionsPipeline`**: Pipeline for preparing data collections
- **`DatamodelsNQPipeline`**: Natural Questions specific pipeline
- **`DatamodelsIndexBasedNQPipeline`**: Index-based Natural Questions pipeline

#### Setters (`setter/`)
Data setting strategies for training datamodels:
- **`BaseSetter`**: Abstract base for data setters
- **`NaiveSetter`**: Simple data setting strategy
- **`StratifiedSetter`**: Stratified sampling approach
- **`IndexBasedSetter`**: Index-based data setting

#### Evaluators (`evaluator/`)
- **`BaseDatamodelsEvaluator`**: Base class for datamodel evaluation
- **`LinearRegressorEvaluator`**: Evaluator for linear regression datamodels

### 3. Models (`models/`)

LLM interface implementations for various model types.

- **`BaseLLM`**: Abstract base class for all LLM implementations
- **`GenericInstructModelHF`**: Hugging Face instruction-following models
- **`GenericInstructBatchHF`**: Batch processing for Hugging Face models
- **`GenericVLLMBatch`**: VLLM batch processing interface
- **`BatchModel`**: Generic batch processing model

### 4. Pipelines (`pipelines/`)

End-to-end processing workflows for different use cases.

- **`BasePipeline`**: Abstract base for all pipelines
- **`BaselinePipeline`**: Basic baseline implementation
- **`RAGPipeline`**: Retrieval-Augmented Generation pipeline

### 5. Evaluators (`evaluators/`)

Comprehensive evaluation framework for model performance assessment.

- **`BaseReferenceEvaluator`**: Base class for reference-based evaluation
- **`BaseUnsupervisedEvaluator`**: Base class for unsupervised evaluation
- **`GenericEvaluator`**: General-purpose evaluator
- **`GleuEvaluator`**: GLEU score evaluation
- **`JudgeEvaluator`**: LLM-as-judge evaluation
- **`Rouge_L_evaluator`**: ROUGE-L score evaluation
- **`Squadv2Evaluator`**: SQuAD v2 specific evaluation

### 6. Vector Stores (`vector_stores/`)

Document storage and retrieval systems.

- **`BaseVectorStore`**: Abstract base for vector storage
- **`Chroma`**: ChromaDB integration for vector storage

### 7. Retrievers (`retrievers/`)

Information retrieval components.

- **`BM25`**: BM25 retrieval implementation

### 8. Utilities (`utils/`)

Supporting tools and configurations.

- **`baseline_config.py`**: Configuration for baseline experiments
- **`experiment_samplers.py`**: Sampling utilities for experiments
- **`score_extractor.py`**: Score extraction utilities
- **`test_utils.py`**: Testing utilities
- **`utils.py`**: General utility functions

## Installation

The DMCR package requires Python 3.11+ and can be installed using pip:

```bash
pip install dmcr
```

### Dependencies

Key dependencies include:
- **PyTorch**: For neural network implementations
- **Transformers**: For Hugging Face model integration
- **LangChain**: For LLM orchestration
- **scikit-learn**: For machine learning utilities
- **VLLM**: For efficient LLM inference
- **Pandas/Polars**: For data manipulation

## Quick Start

### Basic Datamodel Training

```python
from dmcr.datamodels.config import DatamodelConfig
from dmcr.datamodels.pipeline import DatamodelPipeline

# Configure the datamodel
config = DatamodelConfig(
    k=5,
    num_models=10,
    datamodels_path="./models"
)

# Create and run pipeline
pipeline = DatamodelPipeline(config)
# pipeline.train()  # Train the datamodels
```

### Using with Custom Evaluators

```python
from dmcr.evaluators import GenericEvaluator

evaluator = GenericEvaluator()
results = evaluator.evaluate(predictions, references)
```

### Custom Pipeline Implementation

```python
from dmcr.pipelines import BasePipeline
import pandas as pd

class CustomPipeline(BasePipeline):
    def run(self, input: str, k: int) -> str:
        # Custom implementation
        return processed_output
    
    def run_tests(self, data: pd.DataFrame, checkpoint: int, k: int) -> None:
        # Testing implementation
        pass
```

## API Reference

### Configuration Classes

#### `DatamodelConfig`
```python
@dataclass
class DatamodelConfig:
    k: int                    # Number of examples for in-context learning
    num_models: int          # Number of datamodels to train
    datamodels_path: str     # Path to save/load datamodels
```

#### `MemMapConfig`
```python
@dataclass
class MemMapConfig:
    filename: str           # Memory map file name
    dtype: type            # Data type for memory mapping
    shape: tuple           # Shape of the memory mapped array
    mode: str              # Access mode ('r', 'w+', etc.)
```

### Base Classes

All major components extend from abstract base classes:

- **`BaseDataloader`**: For custom data loading implementations
- **`BaseLLM`**: For custom LLM integrations
- **`BasePipeline`**: For custom processing workflows
- **`BaseReferenceEvaluator`**: For custom evaluation metrics
- **`BaseVectorStore`**: For custom vector storage backends

### Pipeline Workflows

The package supports various pipeline workflows:

1. **Training Pipeline**: `TrainModelsPipeline` for training multiple datamodels
2. **Inference Pipeline**: `DatamodelPipeline` for making predictions
3. **Evaluation Pipeline**: Integration with evaluators for performance assessment
4. **RAG Pipeline**: `RAGPipeline` for retrieval-augmented generation

## Contributing

When extending the DMCR package:

1. Follow the abstract base class patterns
2. Implement required abstract methods
3. Use the configuration system for parameterization
4. Add appropriate evaluation metrics
5. Include comprehensive documentation

## License

This project is licensed under the GPL-3.0-or-later license.

## Research Context

This package is designed for research in datamodels for in-context learning, specifically focusing on context reduction techniques while maintaining model performance across multiple tasks. The modular architecture allows researchers to experiment with different components and configurations.