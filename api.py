from typing import List, Dict
from bson import ObjectId
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, File, UploadFile, HTTPException
from pymongo import MongoClient
import os
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json

# Load environment variables
load_dotenv()

# Get API keys and MongoDB URI
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Hugging Face API key not found. Please set API_KEY in the .env file.")

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("Mongo URI not found. Please set MONGO_URI in the .env file.")

# Authenticate with Hugging Face
login(token=api_key)

# Load the Hugging Face model
MODEL_IDENTIFIER = 'SaaraKaizer/contributor_selection'
model = SentenceTransformer(MODEL_IDENTIFIER)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://your-react-app-url.com"  # Production frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB client
client = MongoClient(mongo_uri)
db = client["test"]
collection = db["pads"]
contributors_collection = db["contributors"]

@app.get("/profiles")
async def get_profiles():
    profiles = list(contributors_collection.find({}, {"_id": 0, "name": 1, "about": 1, "bio": 1, "position": 1}))
    return profiles

# Contributor Model
class Contributor(BaseModel):
    name: str
    email: str
    affiliation: str = ""
    position: str = ""
    about: str = ""
    bio: str = ""

@app.post("/contributors")
async def create_contributor(
    name: str = Form(...),
    email: str = Form(...),
    affiliation: str = Form(""),
    position: str = Form(""),
    about: str = Form(""),
    bio: str = Form(""),
    profile_picture: UploadFile = File(None)
):
    contributor_data = {
        "name": name,
        "email": email,
        "affiliation": affiliation,
        "position": position,
        "about": about,
        "bio": bio,
        "profile_picture": profile_picture.filename if profile_picture else None
    }
    inserted_id = contributors_collection.insert_one(contributor_data).inserted_id
    return {"message": "Contributor saved successfully", "id": str(inserted_id)}

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            pad_id = request_data.get("pad_id")

            if not pad_id:
                await websocket.send_text(json.dumps({"error": "Pad ID is required"}))
                continue

            pad_data = collection.find_one({"_id": ObjectId(pad_id)})

            if not pad_data:
                await websocket.send_text(json.dumps({"error": "Pad not found"}))
                continue

            await websocket.send_text(json.dumps(pad_data, default=str))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Input Data Schema
class PadRequest(BaseModel):
    pad_id: str

class SectionResult(BaseModel):
    section_name: str
    best_contributor: Dict[str, str]

class PadProcessingResponse(BaseModel):
    pad_id: str
    results: List[SectionResult]

class Request(BaseModel):
    text: str

class ResponseData(BaseModel):
    name:str
    about: str
    bio: str
    position: str
    score: float

class Candidate(BaseModel):
    full_name: str
    about: str
    bio: str
    position: str

class RequestBody(BaseModel):
    keywords: str
    candidates: List[Candidate]

@app.post("/predict", response_model=ResponseData)
def predict_contributor(body: RequestBody):
    # Extract keywords and candidates
    keywords = body.keywords
    candidates = body.candidates

    # Encode the keyword
    keyword_embedding = model.encode(keywords, convert_to_tensor=True)

    # Prepare the candidate details in the specified format
    candidate_texts = [f"{candidate.full_name}, {candidate.about}, {candidate.bio}, {candidate.position}" for candidate in candidates]
    candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

    # Calculare cosine similarities
    similarity_scores = util.pytorch_cos_sim(keyword_embedding, candidate_embeddings)[0]
    best_match_idx = similarity_scores.argmax().item()

    # Get the best candidate's information
    best_candidate = candidates[best_match_idx]
    return {
        "name": best_candidate.full_name,
        "about": best_candidate.about,
        "bio": best_candidate.bio,
        "position": best_candidate.position,
        "score": similarity_scores[best_match_idx].item()
    }

@app.post("/process_pad")
def process_pad(body: PadRequest):
    pad_id = body.pad_id

    # Get pad data
    pad_data = collection.find_one({"_id": ObjectId(pad_id)})
    if not pad_data:
        return {"message": "Pad not found with ID: {pad_id}"}

    results = []
    # Get authors
    authors = pad_data.get("authors", [])
    # change this accordingly if candidate values change
    candidates = [
        Candidate(full_name=author.get("name", ""),
                  about=author.get("about", ""),
                  bio=author.get("bio", ""),
                  position=author.get("position", "")) # change to affiliation if needed
        for author in authors
    ]

    # Get abstract
    abstract = pad_data.get("abstract", "")
    if abstract:
        abstract_embedding = model.encode(abstract, convert_to_tensor=True)
        candidate_texts = [f"{c.full_name}, {c.about}, {c.bio}, {c.position}" for c in candidates]
        candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(abstract_embedding, candidate_embeddings)[0]
        best_match_idx = similarity_scores.argmax().item()
        best_candidate = candidates[best_match_idx]
        best_contributor = {
            "name": best_candidate.full_name,
            "about": best_candidate.about,
            "bio": best_candidate.bio,
            "position": best_candidate.position,
            "score": str(similarity_scores[best_match_idx].item())
        }
        results.append(SectionResult(section_name="Abstract", best_contributor=best_contributor))

    # Get sections
    sections = pad_data.get("sections", [])
    for section in sections:
        section_content = section.get("content", {}).get("ops", "") #  section_content = section.get("content", {}).get("ops", []), ", ".join([op.get("insert", "") for op in section_content])
        section_embedding = model.encode(section_content, convert_to_tensor=True)

        # Prepare candidate embeddings
        candidate_texts = [f"{c.full_name}, {c.about}, {c.bio}, {c.position}" for c in candidates]
        candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

        # Calculare cosine similarities
        similarity_scores = util.pytorch_cos_sim(section_embedding, candidate_embeddings)[0]
        best_match_idx = similarity_scores.argmax().item()
        best_candidate = candidates[best_match_idx]

        # Save best contributor in MongoDB
        best_contributor = {
            "name": best_candidate.full_name,
            "about": best_candidate.about,
            "bio": best_candidate.bio,
            "position": best_candidate.position,
            "score": str(similarity_scores[best_match_idx].item())
        }

        #section["best_contributor"] = best_contributor

        results.append(SectionResult(section_name=section.get("title", ""), best_contributor=best_contributor))

    contributors_collection.insert_one({
        "pad_id": str(pad_id),
        "sections": [result.dict() for result in results]
    })
    return PadProcessingResponse(pad_id=pad_id, results=results)

@app.get("/")
def root():
    return {"message": "Welcome to the Candidate Selection API"}

@app.post("/updateKeywords")
async def update_keywords(data: dict):
    pad_id = data.get("pad_id")
    section = data.get("section")
    keywords = data.get("keywords")

    if not pad_id or not section or keywords is None:
        return {"error": "Missing required parameters"}

    collection.update_one(
        {"_id": ObjectId(pad_id)},
        {"$set": {f"sections.{section}.keywords": keywords}},
        upsert=True
    )
    return {"message": "Keywords updated successfully"}

@app.post("/updateCandidates")
async def update_candidates(data: dict):
    pad_id = data.get("pad_id")
    section = data.get("section")
    candidates = data.get("candidates")

    if not pad_id or not section or candidates is None:
        return {"error": "Missing required parameters"}

    collection.update_one(
        {"_id": ObjectId(pad_id)},
        {"$set": {f"sections.{section}.candidates": candidates}},
        upsert=True
    )
    return {"message": "Candidates updated successfully"}

class UpdateBestContributorRequest(BaseModel):
    section: str
    bestContributor: Dict[str, str]

@app.post("/updateBestContributor")
async def update_best_contributor(request: UpdateBestContributorRequest):
    pad_id = "your_pad_id_here"  # Replace with the actual pad ID or pass it in the request
    section_title = request.section
    best_contributor = request.bestContributor

    # Find the pad document
    pad_data = collection.find_one({"_id": ObjectId(pad_id)})
    if not pad_data:
        raise HTTPException(status_code=404, detail="Pad not found")

    # Update the best contributor for the specified section
    for section in pad_data.get("sections", []):
        if section.get("title") == section_title:
            section["best_contributor"] = best_contributor
            break
    else:
        raise HTTPException(status_code=404, detail="Section not found")

    # Save the updated pad document back to the database
    collection.update_one({"_id": ObjectId(pad_id)}, {"$set": {"sections": pad_data["sections"]}})

    return {"message": "Best contributor updated successfully"}
