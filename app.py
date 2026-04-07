import streamlit as st
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import heapq
import collections

st.set_page_config(page_title="DCT + Huffman Compression", layout="wide")

st.title("📸 Image Compression using DCT + Huffman Coding")

# -------- QUALITY SLIDER --------
quality = st.slider("🎚 Compression Quality", 10, 100, 50)

# -------- UPLOAD --------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

# -------- HUFFMAN CODING --------
class Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman(data):
    freq = collections.Counter(data)
    heap = [Node(freq[s], s) for s in freq]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(n1.freq + n2.freq, None, n1, n2)
        heapq.heappush(heap, merged)

    return heap[0]

def generate_codes(node, prefix="", codebook={}):
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    generate_codes(node.left, prefix + "0", codebook)
    generate_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_size(data):
    flat = data.flatten().astype(int)
    tree = build_huffman(flat)
    codes = generate_codes(tree)
    total_bits = sum(len(codes[val]) for val in flat)
    return total_bits / 8  # bytes

# -------- JPEG MATRIX --------
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

# -------- PROCESS --------
if uploaded_file is not None:

    file_bytes = uploaded_file.read()
    np_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, channels="BGR", caption="Original Image")

    img = np.float32(img)

    h, w, _ = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w]

    b, g, r = cv2.split(img)

    Q_scaled = Q * (100 / quality)

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

    # Compress
    b_c, g_c, r_c = compress(b), compress(g), compress(r)

    # Huffman Size
    huff_size = (
        huffman_size(b_c) +
        huffman_size(g_c) +
        huffman_size(r_c)
    )

    # Reconstruct
    b_r, g_r, r_r = decompress(b_c), decompress(g_c), decompress(r_c)
    reconstructed = cv2.merge([b_r, g_r, r_r])
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    with col2:
        st.image(reconstructed, channels="BGR", caption="Compressed Image")

    # -------- METRICS --------
    original = img.astype(np.uint8)

    mse = np.mean((original - reconstructed) ** 2)
    psnr = 10 * np.log10((255**2) / mse)
    mae = np.mean(np.abs(original - reconstructed))

    st.subheader("📊 Metrics")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"PSNR: {psnr:.2f} dB")
    st.write(f"MAE: {mae:.2f}")

    # -------- SIZE --------
    original_size = len(file_bytes)
    compressed_size = huff_size

    st.subheader("📦 Size Comparison")
    st.write(f"Original Size: {original_size/1024:.2f} KB")
    st.write(f"Compressed Size (Huffman): {compressed_size/1024:.2f} KB")

    # -------- INSIGHT --------
    st.info("""
    Huffman coding reduces size by encoding repeated values efficiently.
    It works best on images with high redundancy like PNG.
    """)

    # -------- DOWNLOAD --------
    _, buffer = cv2.imencode('.jpg', reconstructed)
    st.download_button(
        "📥 Download Image",
        buffer.tobytes(),
        "compressed.jpg"
    )
