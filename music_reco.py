import streamlit as st
from deepface import DeepFace
import numpy as np
from googleapiclient.discovery import build
from PIL import Image
import cv2

# Replace with your YouTube API key
YOUTUBE_API_KEY = "AIzaSyChob6xQtjYXdnBEPO44dyVn7fCrlpViBA"

# Emotion to base keyword mapping
emotion_to_keyword = {
    "happy": "happy upbeat songs",
    "sad": "sad emotional songs",
    "angry": "intense angry music",
    "surprise": "energetic pop songs",
    "neutral": "chill relaxing music",
    "fear": "dark ambient music",
    "disgust": "heavy metal songs"
}

# Available genres/categories for filtering
music_genres = [
    "pop", "rock", "hip hop", "jazz", "classical", "electronic", "country", "reggae", "metal", "blues"
]

# Year range options for filtering (used as keywords)
year_ranges = {
    "All years": "",
    "2000-2010": "2000 2010",
    "2010-2020": "2010 2020",
    "2020-2025": "2020 2025"
}

def youtube_search_videos(api_key, query, max_results=10):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video',
        videoCategoryId='10'  # Music category
    )
    response = request.execute()
    videos = []
    for item in response.get('items', []):
        videos.append({
            'title': item['snippet']['title'],
            'video_id': item['id']['videoId'],
            'thumbnail': item['snippet']['thumbnails']['high']['url'],
            'channel': item['snippet']['channelTitle']
        })
    return videos

def enhance_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing to improve face detection:
    Convert to grayscale and apply histogram equalization.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(gray)
    # Convert back to 3-channel image expected by DeepFace
    enhanced = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
    return enhanced

st.title("🎵 Emotion-Based Music Playlist Generator with YouTube")

st.markdown("""
Upload your selfie or take a live photo, and get a YouTube music playlist based on your detected emotion.
""")

# Input method
choice = st.radio("Choose input method:", ("Upload Selfie", "Use Webcam"))

img = None
dominant_emotion = None
emotion_probs = None

if choice == "Upload Selfie":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        img = np.array(img.convert('RGB'))

elif choice == "Use Webcam":
    picture = st.camera_input("Take a selfie")
    if picture is not None:
        img = Image.open(picture)
        st.image(img, caption="Captured Image", use_container_width=True)
        img = np.array(img.convert('RGB'))

if img is not None:
    with st.spinner("Enhancing image and analyzing emotion..."):
        try:
            # Enhance image for better face detection
            enhanced_img = enhance_image_for_detection(img)
            
            # Analyze with enforce_detection=False to avoid crash if no face
            analysis = DeepFace.analyze(enhanced_img, actions=['emotion'], enforce_detection=False)

            # DeepFace returns list if multiple faces or dict if one face
            if isinstance(analysis, list):
                analysis = analysis[0]

            if 'dominant_emotion' in analysis and analysis['dominant_emotion']:
                dominant_emotion = analysis['dominant_emotion']
                emotion_probs = analysis['emotion']
                st.success(f"Detected emotion: **{dominant_emotion.capitalize()}**")

                # Show emotion probabilities
                st.subheader("Emotion confidence scores:")
                emotion_display = {k.capitalize(): f"{v:.2f}%" for k, v in emotion_probs.items()}
                st.write(emotion_display)

            else:
                st.warning("Face not detected properly. Please select your emotion manually below.")
                dominant_emotion = None

        except Exception as e:
            st.error(f"😞 Could not analyze emotion.\nError details: {e}")
            dominant_emotion = None

# Manual override of emotion even if detected
if dominant_emotion:
    override = st.checkbox("Override detected emotion and select manually?")
    if override:
        emotions_list = list(emotion_to_keyword.keys())
        manual_emotion = st.selectbox("Select your emotion:", emotions_list, index=emotions_list.index(dominant_emotion.lower()) if dominant_emotion.lower() in emotions_list else 0)
        if manual_emotion:
            dominant_emotion = manual_emotion

# If no dominant emotion detected or overridden, force manual selection
if dominant_emotion is None:
    st.warning("Please select your current emotion manually.")
    emotions_list = list(emotion_to_keyword.keys())
    manual_emotion = st.selectbox("Select your emotion:", emotions_list)
    if manual_emotion:
        dominant_emotion = manual_emotion

if dominant_emotion:
    # Additional user inputs for customization
    st.subheader("Customize your playlist")

    # Number of videos
    num_videos = st.slider("Number of videos to fetch", min_value=5, max_value=20, value=10, step=1)

    # Year range filter
    year_range = st.selectbox("Select music year range", list(year_ranges.keys()), index=0)

    # Music genre/category filter
    genre = st.selectbox("Select music genre/category", ["All genres"] + music_genres, index=0)

    # Construct search query
    base_query = emotion_to_keyword.get(dominant_emotion.lower(), "pop music")
    year_filter = year_ranges.get(year_range, "")
    genre_filter = genre if genre != "All genres" else ""

    # Combine parts, ignoring empty strings
    search_parts = [base_query]
    if genre_filter:
        search_parts.append(genre_filter)
    if year_filter:
        search_parts.append(year_filter)
    final_query = " ".join(search_parts)

    st.info(f"Searching YouTube playlist for: **{final_query}**")

    # Use session_state to hold last query params & results so we can refresh
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'videos' not in st.session_state:
        st.session_state.videos = []

    def fetch_and_store_videos():
        st.session_state.videos = youtube_search_videos(YOUTUBE_API_KEY, final_query, max_results=num_videos)
        st.session_state.last_query = (dominant_emotion, num_videos, year_range, genre)

    generate_clicked = st.button("Generate Playlist")

    if generate_clicked:
        fetch_and_store_videos()

    if st.session_state.videos:
        refresh_clicked = st.button("Refresh Playlist")
        if refresh_clicked:
            fetch_and_store_videos()

    videos = st.session_state.videos

    if videos:
        st.subheader(f"🎶 Playlist for '{dominant_emotion.capitalize()}'")

        for vid in videos:
            st.markdown(f"**{vid['title']}**  \n_Channel: {vid['channel']}_")
            st.image(vid['thumbnail'], width=320)
            video_url = f"https://www.youtube.com/watch?v={vid['video_id']}"
            st.video(video_url)
            st.write("---")
    else:
        if st.session_state.last_query is not None:
            st.warning("No videos found for the current query.")

st.markdown("---")
st.markdown("Made with ❤️ by [Your Name]")
