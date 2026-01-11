"""
API module for the AI Safety Guard
Provides both synchronous and asynchronous interfaces for safety classification
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging
from .services.safety_classifier import SafetyClassifier


class ClassificationRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5


class BatchClassificationRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5


class ClassificationResponse(BaseModel):
    text: str
    is_safe: bool
    confidence: float
    prediction: str
    class_id: int
    blocked: bool


class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]


def create_app():
    app = FastAPI(
        title="AI Safety Guard API",
        description="A content safety classifier to detect potentially harmful text prompts",
        version="1.0.0"
    )
    
    # Initialize the safety classifier
    classifier = SafetyClassifier()
    
    @app.get("/")
    async def root():
        return {"message": "AI Safety Guard API is running"}
    
    @app.post("/classify", response_model=ClassificationResponse)
    async def classify(request: ClassificationRequest):
        try:
            result = classifier.classify(request.text)
            
            # Determine if content should be blocked based on confidence and threshold
            blocked = not result["is_safe"] and result["confidence"] >= request.threshold
            
            return ClassificationResponse(
                text=result["text"],
                is_safe=result["is_safe"],
                confidence=result["confidence"],
                prediction=result["prediction"],
                class_id=result["class_id"],
                blocked=blocked
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/classify/batch", response_model=BatchClassificationResponse)
    async def classify_batch(request: BatchClassificationRequest):
        try:
            results = classifier.classify_batch(request.texts)
            
            # Process results and add blocked flag
            processed_results = []
            for result in results:
                blocked = not result["is_safe"] and result["confidence"] >= request.threshold
                processed_results.append(
                    ClassificationResponse(
                        text=result["text"],
                        is_safe=result["is_safe"],
                        confidence=result["confidence"],
                        prediction=result["prediction"],
                        class_id=result["class_id"],
                        blocked=blocked
                    )
                )
            
            return BatchClassificationResponse(results=processed_results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# For standalone usage
app = create_app()