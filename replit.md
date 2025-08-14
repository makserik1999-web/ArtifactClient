# Artifact Identification System

## Overview

This is a FastAPI-based archaeological artifact identification system that uses similarity scoring algorithms to identify artifacts based on their physical characteristics and location data. The system compares input artifacts against a reference database of known archaeological artifacts, calculating confidence scores based on dimensional, material, and geographical similarities to provide the most likely match.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Framework
- **FastAPI**: Chosen as the web framework for its automatic API documentation, type validation, and high performance
- **Pydantic Models**: Used for request/response validation and serialization, ensuring data integrity and providing clear API schemas

### Data Processing Architecture
- **Similarity Scoring Engine**: Core algorithm that compares input artifacts against reference database using multiple weighted factors
- **Reference Database**: JSON-based artifact database containing physical dimensions, materials, geographical coordinates, and historical context
- **Input Validation**: Comprehensive validation for physical measurements, coordinates, and categorical data

### API Design
- **RESTful Endpoints**: Clean HTTP API following REST principles
- **Structured Request/Response**: Well-defined Pydantic models for consistent data exchange
- **Error Handling**: Proper HTTP status codes and error messages for various failure scenarios

### Data Storage
- **JSON File Storage**: Lightweight file-based storage for artifact reference data, suitable for the current scale
- **In-Memory Processing**: Artifacts loaded into memory for fast comparison operations

### Similarity Algorithm
- **Multi-Factor Scoring**: Combines dimensional, material, geographical, and categorical similarities
- **Weighted Scoring System**: Different factors have different weights in the final confidence calculation
- **Distance Calculations**: Geographical proximity using coordinate-based distance formulas

## External Dependencies

### Core Framework Dependencies
- **FastAPI**: Web framework for building the API
- **Pydantic**: Data validation and serialization library
- **Uvicorn**: ASGI server for running the FastAPI application (implied)

### Python Standard Library
- **json**: For parsing the artifact reference database
- **math**: For geographical distance calculations and similarity scoring
- **os/pathlib**: For file path management and configuration

### Data Sources
- **artifacts.json**: Reference database containing known archaeological artifacts with their physical properties, historical periods, and geographical origins

### Potential Future Integrations
- **Database System**: Could be expanded to use PostgreSQL or MongoDB for larger datasets
- **Machine Learning Libraries**: Could integrate scikit-learn or TensorFlow for more sophisticated classification
- **External Archaeological APIs**: Could connect to museum or archaeological databases for expanded reference data