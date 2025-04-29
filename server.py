from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles





app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

