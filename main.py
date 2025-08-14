from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Artifact Identification System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 2rem auto;
            max-width: 1200px;
        }
        .header {
            background: linear-gradient(45deg, #2c3e50, #34495e);
            color: white;
            padding: 2rem;
            border-radius: 15px 15px 0 0;
            text-align: center;
        }
        .form-section {
            padding: 2rem;
        }
        .results-section {
            padding: 2rem;
            background: #f8f9fa;
            border-radius: 0 0 15px 15px;
            display: none;
        }
        .artifact-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
        }
        .confidence-bar {
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(45deg, #28a745, #20c997);
            transition: width 0.5s ease;
        }
        .score-breakdown {
            display: flex;
            gap: 0.5rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }
        .score-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        .spinner {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .form-control:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        }
        .btn-analyze {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn-analyze:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="main-container">
            <div class="header">
                <h1><i class="fas fa-search"></i> Archaeological Artifact Identification</h1>
                <p class="mb-0">Discover the history behind your archaeological finds using AI-powered similarity analysis</p>
                <small class="text-light">Powered by advanced haversine distance calculations and multi-factor scoring</small>
            </div>
            
            <div class="form-section">
                <form id="artifactForm">
                    <div class="row">
                        <div class="col-md-4">
                            <h5><i class="fas fa-ruler-combined text-primary"></i> Physical Dimensions (cm)</h5>
                            <div class="mb-3">
                                <label for="length" class="form-label">Length</label>
                                <input type="number" class="form-control" id="length" step="0.1" min="0.1" required>
                            </div>
                            <div class="mb-3">
                                <label for="width" class="form-label">Width</label>
                                <input type="number" class="form-control" id="width" step="0.1" min="0.1" required>
                            </div>
                            <div class="mb-3">
                                <label for="height" class="form-label">Height</label>
                                <input type="number" class="form-control" id="height" step="0.1" min="0.1" required>
                            </div>
                        </div>
                        
                        <div class="col-md-4">
                            <h5><i class="fas fa-palette text-warning"></i> Physical Properties</h5>
                            <div class="mb-3">
                                <label for="color" class="form-label">Color</label>
                                <input type="text" class="form-control" id="color" placeholder="e.g., brown, #a0522d" required>
                                <small class="form-text text-muted">Color name or hex code</small>
                            </div>
                            <div class="mb-3">
                                <label for="material" class="form-label">Material</label>
                                <select class="form-control" id="material" required>
                                    <option value="">Select material...</option>
                                    <option value="clay">Clay/Ceramic</option>
                                    <option value="stone">Stone</option>
                                    <option value="metal">Metal</option>
                                    <option value="bronze">Bronze</option>
                                    <option value="iron">Iron</option>
                                    <option value="gold">Gold</option>
                                    <option value="jade">Jade</option>
                                    <option value="flint">Flint</option>
                                    <option value="bone">Bone</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label for="shape" class="form-label">Shape</label>
                                <select class="form-control" id="shape" required>
                                    <option value="">Select shape...</option>
                                    <option value="oval">Oval</option>
                                    <option value="rectangular">Rectangular</option>
                                    <option value="cylindrical">Cylindrical</option>
                                    <option value="triangular">Triangular</option>
                                    <option value="circular">Circular</option>
                                    <option value="crescent">Crescent</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="col-md-4">
                            <h5><i class="fas fa-map-marker-alt text-success"></i> Discovery Location</h5>
                            <div class="mb-3">
                                <label for="latitude" class="form-label">Latitude</label>
                                <input type="number" class="form-control" id="latitude" step="0.0001" min="-90" max="90" required>
                                <small class="form-text text-muted">-90 to 90 degrees</small>
                            </div>
                            <div class="mb-3">
                                <label for="longitude" class="form-label">Longitude</label>
                                <input type="number" class="form-control" id="longitude" step="0.0001" min="-180" max="180" required>
                                <small class="form-text text-muted">-180 to 180 degrees</small>
                            </div>
                            <div class="mt-4">
                                <button type="submit" class="btn btn-primary btn-analyze w-100">
                                    <i class="fas fa-search"></i> Analyze Artifact
                                </button>
                            </div>
                        </div>
                    </div>
                </form>
                
                <div class="loading" id="loading">
                    <i class="fas fa-spinner spinner fa-3x text-primary"></i>
                    <p class="mt-3">Analyzing your artifact using AI similarity matching...</p>
                </div>
            </div>
            
            <div class="results-section" id="results">
                <h4><i class="fas fa-trophy text-warning"></i> Analysis Results</h4>
                <div id="resultsContent"></div>
                
                <div class="mt-4">
                    <a href="/docs" class="btn btn-outline-secondary">
                        <i class="fas fa-code"></i> API Documentation
                    </a>
                    <button class="btn btn-outline-primary ms-2" onclick="resetForm()">
                        <i class="fas fa-redo"></i> Analyze Another
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('artifactForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading state
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            // Collect form data
            const formData = {
                length: parseFloat(document.getElementById('length').value),
                width: parseFloat(document.getElementById('width').value),
                height: parseFloat(document.getElementById('height').value),
                color: document.getElementById('color').value,
                material: document.getElementById('material').value,
                shape: document.getElementById('shape').value,
                latitude: parseFloat(document.getElementById('latitude').value),
                longitude: parseFloat(document.getElementById('longitude').value)
            };
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                displayResults(result);
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('resultsContent').innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle"></i>
                        Error analyzing artifact: ${error.message}
                    </div>
                `;
                document.getElementById('results').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function displayResults(data) {
            const resultsContent = document.getElementById('resultsContent');
            
            let html = `
                <div class="artifact-card">
                    <div class="row">
                        <div class="col-md-8">
                            <h5><i class="fas fa-crown text-warning"></i> Best Match</h5>
                            <h3 class="text-primary">${data.result.artifact}</h3>
                            <p class="text-muted mb-2"><i class="fas fa-history"></i> ${data.result.era}</p>
                            <p class="mb-3">${data.result.reason}</p>
                            
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${data.result.confidence}%"></div>
                            </div>
                            <small class="text-muted">Confidence: ${data.result.confidence}%</small>
                            
                            <div class="score-breakdown">
                                <span class="badge score-badge bg-primary">Size: ${data.result.scores.size.toFixed(1)}</span>
                                <span class="badge score-badge bg-info">Color: ${data.result.scores.color.toFixed(1)}</span>
                                <span class="badge score-badge bg-success">Material: ${data.result.scores.material.toFixed(1)}</span>
                                <span class="badge score-badge bg-warning">Shape: ${data.result.scores.shape.toFixed(1)}</span>
                                <span class="badge score-badge bg-secondary">Location: ${data.result.scores.location.toFixed(1)}</span>
                            </div>
                        </div>
                        <div class="col-md-4 text-center">
                            <div class="bg-light rounded p-3">
                                <i class="fas fa-image fa-4x text-muted mb-2"></i>
                                <p class="small text-muted">Artifact visualization not available</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            if (data.top_candidates && data.top_candidates.length > 1) {
                html += `
                    <h5 class="mt-4"><i class="fas fa-list"></i> Alternative Matches</h5>
                `;
                
                data.top_candidates.slice(1).forEach((candidate, index) => {
                    html += `
                        <div class="artifact-card">
                            <div class="row align-items-center">
                                <div class="col-md-8">
                                    <h6>${candidate.artifact}</h6>
                                    <small class="text-muted">${candidate.era}</small>
                                    <div class="confidence-bar mt-2" style="height: 10px;">
                                        <div class="confidence-fill" style="width: ${candidate.confidence}%; background: #6c757d;"></div>
                                    </div>
                                </div>
                                <div class="col-md-4 text-end">
                                    <span class="badge bg-secondary">${candidate.confidence.toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    `;
                });
            }
            
            resultsContent.innerHTML = html;
            document.getElementById('results').style.display = 'block';
            
            // Smooth scroll to results
            document.getElementById('results').scrollIntoView({ 
                behavior: 'smooth' 
            });
        }
        
        function resetForm() {
            document.getElementById('artifactForm').reset();
            document.getElementById('results').style.display = 'none';
            document.getElementById('loading').style.display = 'none';
        }
        
        // Example data population for demo
        function loadExample() {
            document.getElementById('length').value = '12.5';
            document.getElementById('width').value = '7.8';
            document.getElementById('height').value = '3.2';
            document.getElementById('color').value = 'brown';
            document.getElementById('material').value = 'clay';
            document.getElementById('shape').value = 'oval';
            document.getElementById('latitude').value = '41.9028';
            document.getElementById('longitude').value = '12.4964';
        }
        
        // Add example button after page load
        window.addEventListener('load', function() {
            const exampleBtn = document.createElement('button');
            exampleBtn.className = 'btn btn-outline-info btn-sm ms-2';
            exampleBtn.innerHTML = '<i class="fas fa-magic"></i> Load Example';
            exampleBtn.onclick = loadExample;
            
            const analyzeBtn = document.querySelector('.btn-analyze');
            analyzeBtn.parentNode.appendChild(exampleBtn);
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the HTML frontend interface"""
    return HTML_TEMPLATE

@app.get("/api")
async def api_info():
    """API information endpoint (alternative to old root)"""
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
