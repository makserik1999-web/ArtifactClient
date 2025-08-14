# Enhanced Artifact Identification System

## Overview

This is a FastAPI-based archaeological artifact identification system that uses advanced similarity scoring algorithms to identify artifacts based on their physical characteristics and location data. The system compares input artifacts against a reference database of known archaeological artifacts, calculating detailed confidence scores and providing top 3 candidate matches with explainable AI components.

## Recent Enhancements (August 2025)

- **Haversine Location Scoring**: Implemented precise geographical distance calculations using the haversine formula
- **Component Score Explainability**: Added detailed breakdown of similarity scores for each matching factor
- **Top-3 Candidate Results**: Returns multiple candidates while maintaining backward compatibility
- **Input Normalization**: Enhanced string field processing with synonym mapping and hex color support
- **Configurable Weights**: Environment-variable configurable scoring weights for different similarity factors
- **Enhanced Error Handling**: Improved validation and error messages following FastAPI best practices
- **CORS Support**: Enabled cross-origin requests for development flexibility

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

## API Usage Example

### Sample POST Request
```json
POST /analyze
{
  "length": 12.8,
  "width": 7.5,
  "height": 3.0,
  "color": "brown",
  "material": "clay",
  "shape": "oval",
  "latitude": 41.9,
  "longitude": 12.5
}
```

### Sample Response (Truncated)
```json
{
  "result": {
    "artifact": "Roman Oil Lamp",
    "era": "Roman Empire (27 BC - 476 AD)",
    "scores": {
      "size": 28.7,
      "color": 0,
      "material": 20,
      "shape": 20,
      "location": 10
    },
    "total_score": 78.7,
    "confidence": 78.7,
    "reason": "Strong match: material match (clay), shape match (oval), location proximity (0.4km)"
  },
  "top_candidates": [
    { "artifact": "Roman Oil Lamp", "confidence": 78.7, ... },
    { "artifact": "Greek Amphora", "confidence": 45.2, ... },
    { "artifact": "Egyptian Canopic Jar", "confidence": 32.1, ... }
  ]
}
```

### Environment Configuration
Set custom scoring weights using:
```bash
export ARTIFACT_WEIGHTS='{"size":25,"color":25,"material":20,"shape":20,"location":10}'
```