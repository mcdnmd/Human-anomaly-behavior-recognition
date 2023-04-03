import uvicorn as uvicorn
from fastapi import FastAPI

from src.api.api import router

app = FastAPI()


if __name__ == "__main__":
    app.include_router(router)
    uvicorn.run(app, host="127.0.0.1", port=8000)
