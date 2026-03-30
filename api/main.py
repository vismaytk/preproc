"""DataPrep Pro API — Application Factory.

Clean app factory that creates the FastAPI app, adds middleware,
and includes routers. No business logic here.
"""

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.core.config import get_settings
from api.core.logging import setup_logging
from api.middleware.logging import RequestLoggingMiddleware
from api.routes import image, tabular, text


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    # Initialize structured logging
    setup_logging()

    # Create FastAPI app
    app = FastAPI(
        title="DataPrep Pro API",
        description="A professional SaaS data preprocessing API for machine learning workflows",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Rate Limiter ---
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # --- CORS (Bug Fix #5: read from config, not hardcoded "*") ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Request Logging Middleware ---
    app.add_middleware(RequestLoggingMiddleware)

    # --- Routers ---
    app.include_router(tabular.router)
    app.include_router(text.router)
    app.include_router(image.router)

    # --- Root Endpoints ---
    @app.get("/", tags=["Health"])
    async def root():
        """Root endpoint returning API information."""
        return {
            "message": "Welcome to DataPrep Pro API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }

    return app


# Create the app instance
app = create_app()
