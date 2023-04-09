import sys

import uvicorn as uvicorn
from fastapi import FastAPI

from src.api import router
from src.ui import create_web_app
from loguru import logger

app = FastAPI()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


if __name__ == "__main__":
    create_web_app()
    app.include_router(router)
    uvicorn.run(app, host="127.0.0.1", port=8000)
