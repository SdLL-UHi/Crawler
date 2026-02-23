# Crawler
Crawler to extract relevant information from the university webpage in Markdown

## Quick Start Guide
The crawler can be executed in two ways:
* Directly as a Python script
* As a Docker container

### Run with Python
1. Clone the repository
2. Create a virtual environment
   `python -m venv venv`
3. Activate the virtual environment
   `source venv/bin/activate`
4. Install dependencies
5. *(Optional)* Edit `config.json` and adjust the desired parameters
6. Run the crawler

### Run with Docker
1. Pull the Docker image
   `docker pull ghcr.io/sdll-uhi/crawler:latest`
2. *(Optional)* Create and configure a `config.json` file
   - Only specify parameters that should override the default values.
3. Create an empty output directory
4. Start the container
    ```bash
    docker run \
    -v <output-folder>:/app/out \
    -v $(pwd)/config.json:/app/config.json \
    ghcr.io/sdll-uhi/crawler:latest
    ```