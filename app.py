import streamlit as st
import cv2
import numpy as np
from scipy.fftpack import dct, idct

st.set_page_config(page_title="JPEG Compression", layout="wide")

st.title("📸 JPEG Image Compression using DCT")
st.info("Upload an image to visualize compression using DCT and quantization")

# Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original Image")
        st.image(img, width=500)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    h, w = gray.shape
    h, w = h - h % 8, w - w % 8
    gray = gray[:h, :w]

    Q = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    compressed = np.zeros_like(gray)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = gray[i:i+8, j:j+8] - 128
            dct_block = dct2(block)
            compressed[i:i+8, j:j+8] = np.round(dct_block / Q)

    reconstructed = np.zeros_like(gray)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = compressed[i:i+8, j:j+8]
            idct_block = idct2(block * Q) + 128
            reconstructed[i:i+8, j:j+8] = idct_block

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    with col2:
        st.subheader("🗜 Compressed Image")
        st.image(reconstructed, width=500)

    # Metrics
    mse = np.mean((gray - reconstructed)**2)
    psnr = 100 if mse == 0 else 10*np.log10((255**2)/mse)

    st.subheader("📊 Metrics")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"PSNR: {psnr:.2f} dB")

    # Download
    _, buffer = cv2.imencode('.jpg', reconstructed)
    st.download_button("📥 Download Compressed Image", buffer.tobytes(), "compressed.jpg")