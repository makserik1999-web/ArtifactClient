from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import math
import os
from pathlib import Path

app = FastAPI(
    title="Artifact Identification System",
    description="A FastAPI-based system for identifying archaeological artifacts using similarity scoring",
    version="1.0.0"
)

class ArtifactInput(BaseModel):
    """Input model for artifact analysis requests"""
    length: float = Field(..., description="Length in cm", gt=0)
    width: float = Field(..., description="Width in cm", gt=0)
    height: float = Field(..., description="Height in cm", gt=0)
    color: str = Field(..., description="Color (lowercase)")
    material: str = Field(..., description="Material (lowercase)")
    shape: str = Field(..., description="Shape (lowercase)")
    latitude: float = Field(..., description="Latitude coordinate", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude coordinate", ge=-180, le=180)

    class Config:
        schema_extra = {
            "example": {
                "length": 15.5,
                "width": 8.2,
                "height": 3.1,
                "color": "brown",
                "material": "clay",
                "shape": "oval",
                "latitude": 40.7128,
                "longitude": -74.0060
            }
        }

class ArtifactResponse(BaseModel):
    """Response model for artifact identification results"""
    artifact: str = Field(..., description="Name of the identified artifact")
    era: str = Field(..., description="Historical era of the artifact")
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    reason: str = Field(..., description="Short explanation of the match")

class ArtifactDatabase:
    """Class to handle artifact reference data loading and management"""
    
    def __init__(self, json_file_path: str = "artifacts.json"):
        self.json_file_path = json_file_path
        self.artifacts = []
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load artifact reference data from JSON file"""
        try:
            if not Path(self.json_file_path).exists():
                raise FileNotFoundError(f"Artifacts file {self.json_file_path} not found")
            
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            if not isinstance(data, list):
                raise ValueError("Artifacts JSON must contain a list of artifacts")
            
            if len(data) == 0:
                raise ValueError("Artifacts JSON file is empty")
            
            # Validate required fields for each artifact
            required_fields = ['name', 'era', 'length', 'width', 'height', 'color', 'material', 'shape', 'latitude', 'longitude']
            for i, artifact in enumerate(data):
                missing_fields = [field for field in required_fields if field not in artifact]
                if missing_fields:
                    raise ValueError(f"Artifact {i} missing required fields: {missing_fields}")
            
            self.artifacts = data
            print(f"Successfully loaded {len(self.artifacts)} artifacts from {self.json_file_path}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=f"Artifact reference file not found: {self.json_file_path}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            raise HTTPException(status_code=500, detail="Invalid JSON format in artifacts file")
        except ValueError as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            print(f"Unexpected error loading artifacts: {e}")
            raise HTTPException(status_code=500, detail="Failed to load artifact reference data")

class SimilarityCalculator:
    """Class to handle similarity scoring calculations"""
    
    @staticmethod
    def calculate_size_similarity(input_artifact: ArtifactInput, reference_artifact: Dict[str, Any]) -> float:
        """
        Calculate size similarity score (max 30 points)
        Subtracts absolute differences in dimensions from maximum score
        """
        max_points = 30.0
        
        # Calculate absolute differences for each dimension
        length_diff = abs(input_artifact.length - reference_artifact['length'])
        width_diff = abs(input_artifact.width - reference_artifact['width'])
        height_diff = abs(input_artifact.height - reference_artifact['height'])
        
        # Sum of all dimensional differences
        total_diff = length_diff + width_diff + height_diff
        
        # Subtract differences from max points, ensuring score doesn't go below 0
        score = max(0, max_points - total_diff)
        
        return score
    
    @staticmethod
    def calculate_exact_match_score(input_value: str, reference_value: str, points: float = 20.0) -> float:
        """
        Calculate exact match score for categorical attributes
        Returns full points for exact match, 0 otherwise
        """
        return points if input_value.lower().strip() == str(reference_value).lower().strip() else 0.0
    
    @staticmethod
    def calculate_location_similarity(input_artifact: ArtifactInput, reference_artifact: Dict[str, Any]) -> float:
        """
        Calculate location similarity score (max 10 points)
        Returns 10 points if coordinates are within 1 degree, 0 otherwise
        """
        lat_diff = abs(input_artifact.latitude - reference_artifact['latitude'])
        lon_diff = abs(input_artifact.longitude - reference_artifact['longitude'])
        
        # Check if both latitude and longitude are within 1 degree
        if lat_diff <= 1.0 and lon_diff <= 1.0:
            return 10.0
        else:
            return 0.0
    
    @staticmethod
    def calculate_total_similarity(input_artifact: ArtifactInput, reference_artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate total similarity score and generate reasoning
        """
        # Calculate individual scores
        size_score = SimilarityCalculator.calculate_size_similarity(input_artifact, reference_artifact)
        color_score = SimilarityCalculator.calculate_exact_match_score(input_artifact.color, reference_artifact['color'])
        material_score = SimilarityCalculator.calculate_exact_match_score(input_artifact.material, reference_artifact['material'])
        shape_score = SimilarityCalculator.calculate_exact_match_score(input_artifact.shape, reference_artifact['shape'])
        location_score = SimilarityCalculator.calculate_location_similarity(input_artifact, reference_artifact)
        
        # Calculate total score
        total_score = size_score + color_score + material_score + shape_score + location_score
        
        # Generate reasoning based on matches
        reasons = []
        if color_score > 0:
            reasons.append(f"color match ({reference_artifact['color']})")
        if material_score > 0:
            reasons.append(f"material match ({reference_artifact['material']})")
        if shape_score > 0:
            reasons.append(f"shape match ({reference_artifact['shape']})")
        if location_score > 0:
            reasons.append("location proximity")
        if size_score > 20:  # Good size similarity
            reasons.append("similar dimensions")
        
        if not reasons:
            reason = "Best available match based on dimensional analysis"
        else:
            reason = "Strong match: " + ", ".join(reasons)
        
        return {
            'total_score': total_score,
            'size_score': size_score,
            'color_score': color_score,
            'material_score': material_score,
            'shape_score': shape_score,
            'location_score': location_score,
            'reason': reason
        }

# Initialize artifact database
artifact_db = ArtifactDatabase()

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Artifact Identification System",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze (POST)",
            "docs": "/docs",
            "artifacts_count": len(artifact_db.artifacts)
        }
    }

@app.get("/artifacts")
async def get_artifacts():
    """Get all reference artifacts (for debugging/testing)"""
    return {
        "count": len(artifact_db.artifacts),
        "artifacts": artifact_db.artifacts
    }

@app.post("/analyze", response_model=ArtifactResponse)
async def analyze_artifact(artifact_input: ArtifactInput):
    """
    Analyze an artifact and return the most similar match from the reference database
    
    The similarity scoring system uses:
    - Size similarity: max 30 points (subtracts dimensional differences)
    - Color match: 20 points for exact match
    - Material match: 20 points for exact match  
    - Shape match: 20 points for exact match
    - Location similarity: 10 points if within 1 degree
    
    Total possible score: 100 points
    """
    try:
        if not artifact_db.artifacts:
            raise HTTPException(status_code=500, detail="No reference artifacts available")
        
        best_match = None
        best_score = -1
        best_details = None
        
        # Calculate similarity for each reference artifact
        for reference_artifact in artifact_db.artifacts:
            similarity_details = SimilarityCalculator.calculate_total_similarity(artifact_input, reference_artifact)
            
            if similarity_details['total_score'] > best_score:
                best_score = similarity_details['total_score']
                best_match = reference_artifact
                best_details = similarity_details
        
        if best_match is None:
            raise HTTPException(status_code=500, detail="No suitable match found")
        
        # Calculate confidence as percentage of maximum possible score (100)
        confidence = (best_score / 100.0) * 100.0
        confidence = round(confidence, 2)
        
        return ArtifactResponse(
            artifact=best_match['name'],
            era=best_match['era'],
            confidence=confidence,
            reason=best_details['reason']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during artifact analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "artifacts_loaded": len(artifact_db.artifacts),
        "timestamp": "2025-08-14"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check if artifacts file exists before starting server
    if not Path("artifacts.json").exists():
        print("Warning: artifacts.json not found. Server will fail to load artifacts.")
    
    print("Starting Artifact Identification System...")
    print("Server will be available at: http://0.0.0.0:8000")
    print("API documentation: http://0.0.0.0:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
