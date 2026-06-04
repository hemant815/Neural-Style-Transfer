import streamlit as st
from torchvision import transforms
from PIL import Image
import uuid
from utils.models import VGGEncoder, Decoder
import torch
from utils.utils import Adain
import io
st.title('STYLEFORGE AI')
device = "mps" if torch.backends.mps.is_available() else 'cpu'

@st.cache_resource
def load_models():
    encoder = VGGEncoder('weights/vgg.pth').to(device)

    decoder = Decoder().to(device)

    decoder.load_state_dict(
        torch.load(
            '/Users/hemantpatidar/Desktop/stylo/experiment/experiment1/decoder15.pth',
            map_location=device
            )
        )


    encoder.eval()
    decoder.eval()

    return encoder, decoder


col1, col2 = st.columns(2)
with col1:
    content_data = st.file_uploader('Upload content image ',type=['jpg','png','jpeg'],accept_multiple_files=False)
    if content_data:
        image = Image.open(content_data)
        image = image.resize((512, 512))
        st.image(image, caption="Preview", use_container_width=True)
with col2:
    style_data = st.file_uploader('Upload style image',type=['jpg','png','jpeg'],accept_multiple_files=False)
    if style_data:
        image = Image.open(style_data)
        image = image.resize((512, 512))
        st.image(image, caption="Preview", use_container_width=True)
alpha = st.slider("Style Strength", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

encoder, decoder = load_models()

def style_transfer(content_data,style_data, encoder, decoder,alpha,device):
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])

    content_data = Image.open(content_data).convert('RGB')
    style_data = Image.open(style_data).convert('RGB')

    content_image = transform(content_data).unsqueeze(0).to(device)
    style_image = transform(style_data).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image)
        style_feats = encoder(style_image)

        stylize_feats = Adain(content_feats[-1],style_feats[-1])
        stylize_feats = (alpha * stylize_feats +(1 - alpha) * content_feats[-1])
        stylized_image = decoder(stylize_feats)

        return stylized_image


    
if content_data and style_data:
    st.header('Result')
    if st.button('Generate'):
        with st.spinner("Generating stylized image..."):
            image = style_transfer(content_data,style_data,encoder,decoder,alpha,device)
            image = image.cpu().squeeze(0)
            image = image.clamp(0, 1)
            image = transforms.ToPILImage()(image)
            st.image(image, caption="Generated", use_container_width=True)

            # Convert PIL image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")

            st.download_button(
                label="💾 Save Image",
                data=buffer.getvalue(),
                file_name="/output/stylized_image.png",
                mime="image/png"
            )