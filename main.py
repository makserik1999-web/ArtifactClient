from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import math
import os
from pathlib import Path

app = FastAPI(
    title="Artifact Identification System",
    description="A FastAPI-based system for identifying archaeological artifacts using similarity scoring",
    version="2.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurable weights for scoring components
DEFAULT_WEIGHTS = {"size": 30, "color": 20, "material": 20, "shape": 20, "location": 10}

def load_weights() -> Dict[str, int]:
    """Load scoring weights from environment variable or use defaults"""
    try:
        weights_str = os.getenv("ARTIFACT_WEIGHTS")
        if weights_str:
            weights = json.loads(weights_str)
            # Validate weights
            if isinstance(weights, dict) and all(isinstance(v, (int, float)) for v in weights.values()):
                return {**DEFAULT_WEIGHTS, **weights}
    except (json.JSONDecodeError, TypeError):
        pass
    return DEFAULT_WEIGHTS.copy()

WEIGHTS = load_weights()

# Normalization mappings
MATERIAL_SYNONYMS = {
    "ceramic": "ceramics",
    "clay": "ceramics",
    "stone": "stone", 
    "metal": "metal",
    "bone": "bone"
}

SHAPE_SYNONYMS = {
    "with handle": "handle",
    "sharp edge": "sharp",
    "rounded": "round",
    "elongated": "oval"
}

# Common hex to color name mappings
HEX_TO_COLOR = {
    "#a0522d": "brown",
    "#8b4513": "brown",
    "#d2691e": "orange",
    "#ffd700": "gold",
    "#c0c0c0": "silver",
    "#808080": "grey",
    "#000000": "black",
    "#ffffff": "white",
    "#ff0000": "red",
    "#00ff00": "green",
    "#0000ff": "blue"
}

def normalize_string_field(value: str, synonyms: Optional[Dict[str, str]] = None) -> str:
    """Normalize string fields: trim, lowercase, apply synonyms"""
    normalized = value.strip().lower()
    if synonyms and normalized in synonyms:
        return synonyms[normalized]
    return normalized

def normalize_color(color: str) -> str:
    """Normalize color: handle hex codes and synonyms"""
    color = color.strip().lower()
    # Check if it's a hex code
    if color.startswith('#') and len(color) == 7:
        return HEX_TO_COLOR.get(color, color)
    return color

class ArtifactInput(BaseModel):
    """Input model for artifact analysis requests"""
    length: float = Field(..., description="Length in cm (must be > 0)", gt=0)
    width: float = Field(..., description="Width in cm (must be > 0)", gt=0)
    height: float = Field(..., description="Height in cm (must be > 0)", gt=0)
    color: str = Field(..., description="Color name or hex code (e.g., 'brown' or '#a0522d')")
    material: str = Field(..., description="Material type (e.g., 'clay', 'stone', 'metal')")
    shape: str = Field(..., description="Shape description (e.g., 'oval', 'rectangular')")
    latitude: float = Field(..., description="Latitude coordinate (must be between -90 and 90)", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude coordinate (must be between -180 and 180)", ge=-180, le=180)

    def normalize_fields(self):
        """Apply normalization to string fields"""
        self.color = normalize_color(self.color)
        self.material = normalize_string_field(self.material, MATERIAL_SYNONYMS)
        self.shape = normalize_string_field(self.shape, SHAPE_SYNONYMS)

    class Config:
        json_schema_extra = {
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

class ComponentScores(BaseModel):
    """Component scores for similarity analysis"""
    size: float = Field(..., description="Size similarity score (0-30)")
    color: float = Field(..., description="Color match score (0-20)")
    material: float = Field(..., description="Material match score (0-20)")
    shape: float = Field(..., description="Shape match score (0-20)")
    location: float = Field(..., description="Location proximity score (0-10)")

class CandidateResult(BaseModel):
    """Individual candidate result with detailed scoring"""
    artifact: str = Field(..., description="Name of the artifact")
    era: str = Field(..., description="Historical era of the artifact")
    scores: ComponentScores = Field(..., description="Detailed component scores")
    total_score: float = Field(..., description="Total similarity score")
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    reason: str = Field(..., description="Short explanation of the match")

class ArtifactResponse(BaseModel):
    """Enhanced response model with backward compatibility and top candidates"""
    result: CandidateResult = Field(..., description="Best matching artifact (for backward compatibility)")
    top_candidates: List[CandidateResult] = Field(..., description="Top 3 candidate matches")

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
                raise HTTPException(status_code=500, detail="Reference DB not loaded")
            
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            if not isinstance(data, list):
                raise HTTPException(status_code=500, detail="Reference DB not loaded")
            
            if len(data) == 0:
                raise HTTPException(status_code=500, detail="Reference DB not loaded")
            
            # Validate required fields for each artifact
            required_fields = ['name', 'era', 'length', 'width', 'height', 'color', 'material', 'shape', 'latitude', 'longitude']
            for i, artifact in enumerate(data):
                missing_fields = [field for field in required_fields if field not in artifact]
                if missing_fields:
                    raise HTTPException(status_code=500, detail="Reference DB not loaded")
            
            self.artifacts = data
            print(f"Successfully loaded {len(self.artifacts)} artifacts from {self.json_file_path}")
            
        except HTTPException:
            raise
        except (FileNotFoundError, json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Error loading artifacts: {e}")
            raise HTTPException(status_code=500, detail="Reference DB not loaded")

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in kilometers
    """
    # Convert decimal degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return c * r

class SimilarityCalculator:
    """Enhanced similarity scoring with configurable weights and haversine distance"""
    
    @staticmethod
    def calculate_size_similarity(input_artifact: ArtifactInput, reference_artifact: Dict[str, Any]) -> float:
        """
        Calculate size similarity score using configurable weights
        Distributes points by penalizing absolute differences across dimensions
        """
        max_points = WEIGHTS["size"]
        
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
    def calculate_exact_match_score(input_value: str, reference_value: str, attribute_type: str) -> float:
        """
        Calculate exact match score for categorical attributes with normalization
        Returns full weight for exact match after normalization, 0 otherwise
        """
        max_points = WEIGHTS[attribute_type]
        
        # Normalize both values for comparison
        if attribute_type == "color":
            input_normalized = normalize_color(input_value)
            ref_normalized = normalize_color(str(reference_value))
        elif attribute_type == "material":
            input_normalized = normalize_string_field(input_value, MATERIAL_SYNONYMS)
            ref_normalized = normalize_string_field(str(reference_value), MATERIAL_SYNONYMS)
        elif attribute_type == "shape":
            input_normalized = normalize_string_field(input_value, SHAPE_SYNONYMS)
            ref_normalized = normalize_string_field(str(reference_value), SHAPE_SYNONYMS)
        else:
            input_normalized = input_value.lower().strip()
            ref_normalized = str(reference_value).lower().strip()
        
        return max_points if input_normalized == ref_normalized else 0.0
    
    @staticmethod
    def calculate_location_similarity(input_artifact: ArtifactInput, reference_artifact: Dict[str, Any]) -> float:
        """
        Calculate location similarity score using haversine distance in kilometers
        Threshold-based scoring: <10km=10pts, <50km=6pts, <150km=3pts, else=0pts
        """
        distance_km = haversine(
            input_artifact.latitude, input_artifact.longitude,
            reference_artifact['latitude'], reference_artifact['longitude']
        )
        
        if distance_km < 10:
            return 10.0
        elif distance_km < 50:
            return 6.0
        elif distance_km < 150:
            return 3.0
        else:
            return 0.0
    
    @staticmethod
    def calculate_total_similarity(input_artifact: ArtifactInput, reference_artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate total similarity score with detailed component breakdown
        """
        # Calculate individual component scores
        size_score = SimilarityCalculator.calculate_size_similarity(input_artifact, reference_artifact)
        color_score = SimilarityCalculator.calculate_exact_match_score(input_artifact.color, reference_artifact['color'], "color")
        material_score = SimilarityCalculator.calculate_exact_match_score(input_artifact.material, reference_artifact['material'], "material")
        shape_score = SimilarityCalculator.calculate_exact_match_score(input_artifact.shape, reference_artifact['shape'], "shape")
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
            distance_km = haversine(
                input_artifact.latitude, input_artifact.longitude,
                reference_artifact['latitude'], reference_artifact['longitude']
            )
            reasons.append(f"location proximity ({distance_km:.1f}km)")
        if size_score > WEIGHTS["size"] * 0.7:  # Good size similarity (>70% of max)
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
        "version": "2.0.0",
        "endpoints": {
            "analyze": "/analyze (POST)",
            "docs": "/docs",
            "artifacts_count": len(artifact_db.artifacts)
        },
        "weights": WEIGHTS
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
    Analyze an artifact and return the most similar matches from the reference database
    
    The enhanced similarity scoring system uses configurable weights:
    - Size similarity: penalizes dimensional differences (default: 30 points max)
    - Color match: exact match after normalization (default: 20 points)
    - Material match: exact match with synonyms (default: 20 points)  
    - Shape match: exact match with synonyms (default: 20 points)
    - Location similarity: haversine distance thresholds (default: 10 points)
    
    Returns both the best match (for backward compatibility) and top 3 candidates.
    """
    try:
        if not artifact_db.artifacts:
            raise HTTPException(status_code=500, detail="Reference DB not loaded")
        
        # Apply input normalization
        artifact_input.normalize_fields()
        
        candidates = []
        
        # Calculate similarity for each reference artifact
        for reference_artifact in artifact_db.artifacts:
            similarity_details = SimilarityCalculator.calculate_total_similarity(artifact_input, reference_artifact)
            
            # Calculate confidence as rounded total score (already 0-100 scale)
            confidence = round(similarity_details['total_score'], 1)
            
            candidate = CandidateResult(
                artifact=reference_artifact['name'],
                era=reference_artifact['era'],
                scores=ComponentScores(
                    size=similarity_details['size_score'],
                    color=similarity_details['color_score'],
                    material=similarity_details['material_score'],
                    shape=similarity_details['shape_score'],
                    location=similarity_details['location_score']
                ),
                total_score=similarity_details['total_score'],
                confidence=confidence,
                reason=similarity_details['reason']
            )
            candidates.append(candidate)
        
        if not candidates:
            raise HTTPException(status_code=500, detail="No suitable match found")
        
        # Sort candidates by total score (descending) and take top 3
        candidates.sort(key=lambda x: x.total_score, reverse=True)
        top_candidates = candidates[:3]
        
        # Best match is the first candidate (for backward compatibility)
        best_match = top_candidates[0]
        
        return ArtifactResponse(
            result=best_match,
            top_candidates=top_candidates
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
        "weights": WEIGHTS,
        "timestamp": "2025-08-14"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check if artifacts file exists before starting server
    if not Path("artifacts.json").exists():
        print("Warning: artifacts.json not found. Server will fail to load artifacts.")
    
    print("Starting Enhanced Artifact Identification System...")
    print("Server will be available at: http://0.0.0.0:8000")
    print("API documentation: http://0.0.0.0:8000/docs")
    print(f"Using weights: {WEIGHTS}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
