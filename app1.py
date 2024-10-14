import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from serpapi import GoogleSearch

# Set up your Streamlit page configuration
st.set_page_config(page_title="Image Caption & Search", layout="centered")

# Title of the app
st.title("Image Caption Generator & Image Search")

# Input field for image URL
image_url = st.text_input("Enter the image URL:")

# If the user has entered a URL, process the image
if image_url:
    # Load pre-trained BLIP model and processor for image captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Define headers for requesting the image
    headers = {
        "User-Agent": "MyBot/1.0 (MyContactInformation@example.com)"
    }
    
    try:
        # Fetch the image from the URL
        response = requests.get(image_url, stream=True, headers=headers)
        response.raise_for_status()  # Raise error for bad response
        
        # Load and display the image
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image using BLIP processor
        inputs = processor(images=img, return_tensors="pt")
        
        # Generate caption using the BLIP model
        outputs = model.generate(**inputs)
        summary = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Display the generated summary
        st.write(f"Generated summary: **{summary}**")
        
        # Input field for user query
        query = st.text_input("Enter your search query:")
        
        if query:
            # Combine the caption and user query for image search
            web_query = summary + " " + query
            
            # SerpAPI key (replace with your own key)
            api_key = "43484dc45cd50b40a199e1da0b4baba474146b822a1645d00976ff7065c572c6"
            
            # Set up search parameters
            params = {
                "q": web_query,  # search query
                "tbm": "isch",  # search type (image search)
                "api_key": api_key  # your SerpAPI key
            }
            
            # Create GoogleSearch object with parameters
            search = GoogleSearch(params)
            
            # Fetch results
            results = search.get_dict()
            
            # Extract and display image URLs from search results
            st.subheader("Search Results:")
            if 'images_results' in results:
                for image_result in results['images_results']:
                    st.image(image_result['original'], caption=image_result['title'], use_column_width=True)
                    st.write(f"Image URL: {image_result['original']}")
            else:
                st.write("No image results found.")
                
    except Exception as e:
        st.error(f"Error fetching the image: {e}")
else:
    st.write("Please enter an image URL to get started.")
