"""
This starts the FastAPI backend defined in `backend/main.py`, so you can run:

    python app.py

to launch the API server on port 8000.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

import uvicorn


def main() -> None:
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    print(
       "Backend starting on http://localhost:8000"
    )
    main()


