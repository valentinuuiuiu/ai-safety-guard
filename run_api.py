"""
Script to run the AI Safety Guard API
"""
import uvicorn
from ai_safety_guard.config import get_config


def main():
    config = get_config()
    print(f"Starting AI Safety Guard API on {config.api_host}:{config.api_port}")
    print(f"Using model: {config.model_name}")
    print("Visit http://localhost:8000/docs for API documentation")
    
    uvicorn.run(
        "ai_safety_guard.api:app",
        host=config.api_host,
        port=config.api_port,
        reload=True  # Enable hot reloading during development
    )


if __name__ == "__main__":
    main()