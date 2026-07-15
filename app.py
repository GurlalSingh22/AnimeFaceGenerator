import time
import torch
import streamlit as st

from inference import generate_faces

# =====================================
# Page Config (Always First)
# =====================================

st.set_page_config(
    page_title="Anime Face Generator",
    page_icon="🎨",
    layout="wide"
)

# =====================================
# Load CSS
# =====================================

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )
    st.image(
    "assets/banner.png",
    use_container_width=True
)
    # =====================================
    # Title
    # =====================================

    st.markdown("""
    <div class="main-title">
    🎨 Anime Face Generator
    </div>

    <div class="sub-title">
    Generate beautiful Anime Faces using your own trained DCGAN Model
    </div>
    """, unsafe_allow_html=True)

    # =====================================
    # Model Information
    # =====================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        

        with col2:
            

            with col3:
                

                with col4:
                    

                    # =====================================
                    # Sidebar
                    # =====================================

                    st.sidebar.header("⚙️ Settings")

                    num_images = st.sidebar.selectbox(
                        "Number of Faces",
                        [1, 4, 16, 64],
                        index=2
                    )

                    st.sidebar.markdown("---")

                    device = "GPU 🚀" if torch.cuda.is_available() else "CPU 💻"

                    st.sidebar.success(f"Running on : {device}")

                    generate = st.sidebar.button("🚀 Generate Faces")

                    # =====================================
                    # Main
                    # =====================================

                    if generate:

                        start = time.time()

                        with st.spinner("Generating Anime Faces..."):
                            image_path = generate_faces(num_images)

                            end = time.time()

                            st.success(f"Generated in {end-start:.2f} sec")

                            st.image(
                                image_path,
                                use_container_width=True
                            )

                            with open(image_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Image",
                                    data=file,
                                    file_name="anime_faces.png",
                                    mime="image/png"
                                )

                                # =====================================
                                # About Model
                                # =====================================

                                with st.expander("📖 About this Model"):

                                    st.write("""
                                    This Anime Face Generator was trained from scratch using a Deep Convolutional GAN (DCGAN).

                                    **Dataset:** Anime Face Dataset

                                    **Resolution:** 128 × 128

                                    **Framework:** PyTorch

                                    **Generator:** ConvTranspose2D

                                    **Discriminator:** Conv2D

                                    **Loss:** BCEWithLogitsLoss

                                    **Optimizer:** Adam
                                    """)

                                    # =====================================
                                    # Footer
                                    # =====================================

                                    st.markdown("---")

                                    st.caption(
                                        "Made with ❤️ by Gurlal Singh | PyTorch • DCGAN • Streamlit"
                                    )