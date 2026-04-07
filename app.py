import streamlit as st
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

st.set_page_config(page_title="JPEG Compression", layout="wide")

st.title("📸 JPEG Image Compression using DCT (Color)")

# -------- QUALITY SLIDER --------
quality = st.slider("🎚 Compression Quality", 10, 100, 50)

# -------- UPLOAD --------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

# JPEG Quantization Matrix
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

# Scale matrix using quality
Q_scaled = Q * (100 / quality)

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# -------- PROCESS --------
if uploaded_file is not None:

    file_bytes = uploaded_file.read()
    np_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original Image")
        st.image(img, channels="BGR", width=500)

    img = np.float32(img)

    h, w, _ = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w]

    b, g, r = cv2.split(img)

    def compress(channel):
        comp = np.zeros_like(channel)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = channel[i:i+8, j:j+8] - 128
                dct_block = dct2(block)
                comp[i:i+8, j:j+8] = np.round(dct_block / Q_scaled)
        return comp

    def decompress(channel):
        rec = np.zeros_like(channel)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = channel[i:i+8, j:j+8]
                idct_block = idct2(block * Q_scaled) + 128
                rec[i:i+8, j:j+8] = idct_block
        return rec

    # Compress & Decompress
    b_c, g_c, r_c = compress(b), compress(g), compress(r)
    b_r, g_r, r_r = decompress(b_c), decompress(g_c), decompress(r_c)

    reconstructed = cv2.merge([b_r, g_r, r_r])
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    with col2:
        st.subheader("🗜 Compressed Image")
        st.image(reconstructed, channels="BGR", width=500)

    original = img.astype(np.uint8)

    # -------- METRICS --------
    mse = np.mean((original - reconstructed) ** 2)
    psnr = 100 if mse == 0 else 10 * np.log10((255**2) / mse)
    mae = np.mean(np.abs(original - reconstructed))

    comp_nonzero = np.count_nonzero(b_c) + np.count_nonzero(g_c) + np.count_nonzero(r_c)
    cr = original.size / comp_nonzero

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("PSNR", f"{psnr:.2f} dB")
    col3.metric("MAE", f"{mae:.2f}")

    st.metric("Compression Ratio", f"{cr:.2f}")

    # -------- REAL FILE SIZE --------
    original_size = len(file_bytes)

    _, buffer = cv2.imencode('.jpg', reconstructed)
    compressed_size = len(buffer)

    st.write(f"📦 Original File Size: {original_size/1024:.2f} KB")
    st.write(f"📦 Compressed File Size: {compressed_size/1024:.2f} KB")

    # -------- DIFFERENCE HEATMAP --------
    diff = cv2.absdiff(original, reconstructed)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    st.subheader("🔍 Difference Heatmap")
    plt.imshow(diff)
    plt.colorbar()
    st.pyplot(plt)

    # -------- DOWNLOAD --------
    st.download_button(
        "📥 Download Compressed Image",
        buffer.tobytes(),
        "compressed.jpg"
    )
