"""ASGI entry point: uvicorn reconlens.api.main:app"""

from reconlens.api.app import create_app

app = create_app()
