from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
