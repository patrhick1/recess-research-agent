import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable for Replit compatibility
    port = int(os.environ.get('PORT', 8000))
    # Run with host='0.0.0.0' for Replit
    uvicorn.run("combined_app:fastapi_app", host="0.0.0.0", port=port, reload=True) 