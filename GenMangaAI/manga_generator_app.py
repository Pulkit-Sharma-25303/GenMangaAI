
import streamlit as st
import cv2
import textwrap
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import os
import random
import io
import base64

# Set page config
st.set_page_config(
    page_title="GenManga AI - Manga Panel Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E3440;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #5E81AC;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3B4252;
        margin: 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, #ECEFF4, #E5E9F0);
        border-radius: 5px;
    }
    .info-box {
        background: #D8DEE9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #5E81AC;
    }
    .stButton > button {
        background: linear-gradient(90deg, #5E81AC, #81A1C1);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #4C7499, #6B8FB5);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'character_db' not in st.session_state:
    st.session_state.character_db = {}
if 'generated_panels' not in st.session_state:
    st.session_state.generated_panels = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Model loading functions
@st.cache_resource
def load_models():
    """Load AI models with caching"""
    try:
        # Load Stable Diffusion
        sd_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP for character detection
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        return sd_model, clip_model, clip_processor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def generate_manga_panel(prompt, style="shonen", expression="neutral", character_description="", seed=None, sd_model=None):
    """Generate a manga panel with given parameters"""
    if sd_model is None:
        st.error("Models not loaded. Please wait for initialization.")
        return None

    # Set seed
    if seed is None:
        seed = random.randint(0, 999999)
    generator = torch.manual_seed(seed)

    # Create full prompt
    full_prompt = f"{character_description}, {prompt}, manga-style, {style}, character expression: {expression}, black and white"

    # Generate image
    with st.spinner("Generating manga panel..."):
        image = sd_model(full_prompt, generator=generator).images[0]

    return image, seed

def detect_and_remember_character(image, database, clip_model, clip_processor):
    """Detect and store character features"""
    try:
        inputs = clip_processor(images=image, return_tensors="pt")
        embedding = clip_model.get_image_features(**inputs)

        # Store in database with timestamp
        import time
        timestamp = str(int(time.time()))
        database[timestamp] = embedding.detach().numpy()

        return True
    except Exception as e:
        st.error(f"Error in character detection: {str(e)}")
        return False

def add_speech_bubble(image, text, position):
    """Add speech bubble to image"""
    draw = ImageDraw.Draw(image)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Text wrapping
    max_chars = 22
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    x, y = position
    bubble_width = 160
    bubble_height = 80

    # Draw bubble
    draw.ellipse([x, y, x + bubble_width, y + bubble_height], 
                fill="white", outline="black", width=3)

    # Add text
    text_x = x + 15
    text_y = y + 30
    draw.text((text_x, text_y), text, fill="black", font=font)

    return image

def create_manga_collage(images, panels_per_row=2, margin=20):
    """Create a collage from multiple panels"""
    if not images:
        return None

    num_images = len(images)

    # Resize all panels to same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    resized_images = [img.resize((max_width, max_height)) for img in images]

    rows = (num_images + panels_per_row - 1) // panels_per_row

    # Calculate collage size
    total_width = panels_per_row * max_width + (panels_per_row + 1) * margin
    total_height = rows * max_height + (rows + 1) * margin

    # Create collage
    collage = Image.new("RGB", (total_width, total_height), "white")

    for idx, img in enumerate(resized_images):
        row = idx // panels_per_row
        col = idx % panels_per_row
        x = margin + col * (max_width + margin)
        y = margin + row * (max_height + margin)
        collage.paste(img, (x, y))

    return collage

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">📚 GenManga AI - Manga Panel Generator</div>', unsafe_allow_html=True)

    # Initialize models
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a few minutes."):
            sd_model, clip_model, clip_processor = load_models()
            if sd_model is not None:
                st.session_state.sd_model = sd_model
                st.session_state.clip_model = clip_model
                st.session_state.clip_processor = clip_processor
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please refresh the page.")
                return

    # Sidebar for controls
    with st.sidebar:
        st.markdown('<div class="section-header">🎨 Generation Controls</div>', unsafe_allow_html=True)

        # Character description
        character_desc = st.text_area(
            "Character Description", 
            value="A one-eyed samurai with long black hair, red tattered kimono, and cursed katana",
            help="Describe your main character to maintain consistency"
        )

        # Style options
        style = st.selectbox(
            "Manga Style",
            ["shonen", "shoujo", "seinen", "josei", "kodomomuke"],
            help="Choose the manga style"
        )

        # Expression options
        expression = st.selectbox(
            "Character Expression",
            ["neutral", "happy", "sad", "angry", "surprised", "serious", "scared", "determined", "pain", "shocked", "relieved"],
            help="Character's facial expression"
        )

        # Seed control
        use_random_seed = st.checkbox("Use Random Seed", value=True)
        if not use_random_seed:
            seed = st.number_input("Seed", value=123456, min_value=0, max_value=999999)
        else:
            seed = None

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Generate Panel", "💬 Add Speech Bubbles", "📖 Create Collage", "🎭 Predefined Story"])

    # Tab 1: Generate Panel
    with tab1:
        st.markdown('<div class="section-header">Generate Manga Panel</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            # Scene prompt
            scene_prompt = st.text_area(
                "Scene Description",
                value="standing on a battlefield at dusk, wind blowing",
                help="Describe the scene you want to generate"
            )

            if st.button("🎨 Generate Panel", type="primary"):
                if st.session_state.models_loaded:
                    image, used_seed = generate_manga_panel(
                        scene_prompt, 
                        style, 
                        expression, 
                        character_desc, 
                        seed, 
                        st.session_state.sd_model
                    )

                    if image:
                        # Store in session state
                        st.session_state.generated_panels.append({
                            'image': image,
                            'prompt': scene_prompt,
                            'seed': used_seed,
                            'style': style,
                            'expression': expression
                        })

                        # Detect character
                        detect_and_remember_character(
                            image, 
                            st.session_state.character_db, 
                            st.session_state.clip_model, 
                            st.session_state.clip_processor
                        )

                        st.success(f"Panel generated! (Seed: {used_seed})")

        with col2:
            # Display latest generated panel
            if st.session_state.generated_panels:
                latest_panel = st.session_state.generated_panels[-1]
                st.image(latest_panel['image'], caption=f"Latest Panel - {latest_panel['prompt']}")

                # Panel info
                st.info(f"""
                **Seed:** {latest_panel['seed']}
                **Style:** {latest_panel['style']}
                **Expression:** {latest_panel['expression']}
                """)

    # Tab 2: Add Speech Bubbles
    with tab2:
        st.markdown('<div class="section-header">Add Speech Bubbles</div>', unsafe_allow_html=True)

        if st.session_state.generated_panels:
            # Select panel
            panel_options = [f"Panel {i+1}: {panel['prompt'][:30]}..." for i, panel in enumerate(st.session_state.generated_panels)]
            selected_panel_idx = st.selectbox("Select Panel", range(len(panel_options)), format_func=lambda x: panel_options[x])

            col1, col2 = st.columns([1, 1])

            with col1:
                # Speech bubble controls
                speech_text = st.text_input("Speech Text", value="Hello, world!")
                bubble_x = st.slider("Bubble X Position", 0, 500, 150)
                bubble_y = st.slider("Bubble Y Position", 0, 500, 100)

                if st.button("💬 Add Speech Bubble"):
                    selected_panel = st.session_state.generated_panels[selected_panel_idx]
                    modified_image = add_speech_bubble(
                        selected_panel['image'].copy(),
                        speech_text,
                        (bubble_x, bubble_y)
                    )

                    # Update the panel
                    st.session_state.generated_panels[selected_panel_idx]['image'] = modified_image
                    st.success("Speech bubble added!")

            with col2:
                # Show selected panel
                if selected_panel_idx < len(st.session_state.generated_panels):
                    st.image(st.session_state.generated_panels[selected_panel_idx]['image'], 
                            caption=f"Panel {selected_panel_idx + 1}")
        else:
            st.info("Generate some panels first!")

    # Tab 3: Create Collage
    with tab3:
        st.markdown('<div class="section-header">Create Manga Collage</div>', unsafe_allow_html=True)

        if st.session_state.generated_panels:
            # Collage settings
            col1, col2 = st.columns([1, 1])

            with col1:
                panels_per_row = st.slider("Panels per Row", 1, 4, 2)
                margin = st.slider("Margin", 10, 50, 20)

                # Panel selection
                st.subheader("Select Panels for Collage")
                selected_panels = []
                for i, panel in enumerate(st.session_state.generated_panels):
                    if st.checkbox(f"Panel {i+1}: {panel['prompt'][:30]}...", key=f"panel_{i}"):
                        selected_panels.append(panel['image'])

                if st.button("📖 Create Collage") and selected_panels:
                    collage = create_manga_collage(selected_panels, panels_per_row, margin)
                    if collage:
                        st.session_state.collage = collage
                        st.success("Collage created!")

            with col2:
                # Display collage
                if hasattr(st.session_state, 'collage'):
                    st.image(st.session_state.collage, caption="Manga Collage")
        else:
            st.info("Generate some panels first!")

    # Tab 4: Predefined Story
    with tab4:
        st.markdown('<div class="section-header">Generate Predefined Samurai Story</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>📚 Predefined Story:</strong> Generate a complete 6-panel samurai story with speech bubbles and create a final collage.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗾 Generate Complete Samurai Story", type="primary"):
            if st.session_state.models_loaded:
                # Clear existing panels
                st.session_state.generated_panels = []

                # Story scenes
                scenes = [
                    ("standing on a battlefield at dusk, wind blowing", "serious", "Tonight... it ends."),
                    ("slashing through an enemy in a bamboo forest", "angry", "I swore on his grave I would finish this!"),
                    ("kneeling by a grave, remembering a fallen comrade", "sad", "You're still chasing ghosts, brother?"),
                    ("blocking a surprise attack in a burning village", "shocked", "HAAAARGH!"),
                    ("shouting in pain after a slash, close-up", "pain", "I won't fall... not yet..."),
                    ("standing victorious in the rain over a defeated enemy", "relieved", "Forgive me.")
                ]

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (scene, expr, speech) in enumerate(scenes):
                    status_text.text(f"Generating panel {i+1}/6: {scene[:30]}...")

                    # Generate panel
                    image, used_seed = generate_manga_panel(
                        scene, 
                        style, 
                        expr, 
                        character_desc, 
                        123456,  # Fixed seed for consistency
                        st.session_state.sd_model
                    )

                    if image:
                        # Add speech bubble
                        bubble_positions = [(150, 60), (100, 200), (180, 150), (200, 100), (160, 220), (180, 100)]
                        image_with_bubble = add_speech_bubble(image.copy(), speech, bubble_positions[i])

                        # Store panel
                        st.session_state.generated_panels.append({
                            'image': image_with_bubble,
                            'prompt': scene,
                            'seed': used_seed,
                            'style': style,
                            'expression': expr
                        })

                        # Detect character
                        detect_and_remember_character(
                            image, 
                            st.session_state.character_db, 
                            st.session_state.clip_model, 
                            st.session_state.clip_processor
                        )

                    progress_bar.progress((i + 1) / 6)

                # Create final collage
                status_text.text("Creating final collage...")
                all_panels = [panel['image'] for panel in st.session_state.generated_panels]
                collage = create_manga_collage(all_panels, 3, 30)

                if collage:
                    st.session_state.collage = collage
                    status_text.text("Story generation complete!")
                    st.success("Complete samurai story generated!")

                    # Display final collage
                    st.image(collage, caption="Complete Samurai Story")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <strong>GenManga AI</strong> - Created with ❤️ using Streamlit, Stable Diffusion, and CLIP
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
