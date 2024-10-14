Intelligent Image-Prompt Based Search Pipeline

Project Overview

This project implements an intelligent pipeline that accepts an image and a user-defined prompt as input. The prompt specifies the user's requirement, such as finding similar objects or variations based on the input image. The pipeline processes both the image and the prompt, then returns relevant web search results that align with the user's specific needs.

Key Features:
Image + Prompt-Based Search: Allows users to submit an image and a text query to specify the object and desired variations for search.
Open-Source Models: Utilizes open-source Vision-Language Models (VLMs) for processing the image and query without reliance on proprietary APIs.
Free-Tier Web Search APIs: Uses free-tier web search APIs to fetch search results based on the input without violating usage restrictions.
Web App for Demonstration: A simple web application demonstrates the functionality of the intelligent pipeline.
Project Structure

bash
Copy code
├── app.py               # Backend code for the web application
├── collab_notebook.ipynb # Colab notebook for testing and development
├── video.mp4            # Demonstration video of the pipeline
├── README.md            # Project documentation
File Descriptions:
app.py:
Contains the backend for the web app (built with Flask or FastAPI).
Implements the main pipeline, which processes the image and text prompt using an open-source Vision-Language Model (VLM).
Fetches relevant web search results from a free-tier web search API.
Provides API endpoints for uploading images, submitting prompts, and retrieving search results.
collab_notebook.ipynb:
A Colab notebook for developing, testing, and fine-tuning the image-prompt pipeline.
Provides code for data preprocessing, loading the model, and querying web search APIs.
Can be used to experiment with the image-prompt pipeline outside of the web app.
video.mp4:
A demonstration video showcasing the use of the web app. It walks through the process of uploading an image, submitting a prompt, and receiving relevant search results.
How It Works

Input:
Image: The user uploads an image of the object.
Prompt: The user provides a text-based query specifying what they are searching for (e.g., similar products, variations of the same item).
Pipeline:
The image and prompt are processed by an open-source Vision-Language Model (VLM) that extracts visual features and contextual understanding.
The system queries a free-tier web search API to retrieve relevant search results based on the input.
Output:
The system returns search results, which are displayed in the web application.
Requirements

Python Libraries:
Flask / FastAPI
OpenCV (for image processing)
PyTorch / TensorFlow (for loading the VLM)
Requests (for API requests)
Free-tier Web Search API (such as DuckDuckGo or other similar services)
Installation
Clone the repository:
bash
Copy code
git clone <repository-url>
Install the required Python packages:
bash
Copy code
pip install -r requirements.txt
Run the web app:
Run the ngrok code on collab notebook
This will give you live publicly available link
Running the Colab Notebook
Open the collab_notebook.ipynb file in Google Colab.
Follow the steps in the notebook to run and test the pipeline using various input images and prompts.
Usage Instructions

Web App:
Run the web app using the command described above.
Upload an image of an object and provide a prompt describing your search requirements.
The app will return web search results based on the input.
Colab Notebook:
Use the notebook to test the pipeline by providing an image and a prompt.
The notebook will execute the search and return the relevant results.
Demonstration Video

A demonstration video (video.mp4) is included in the repository to showcase the working pipeline.

