import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import load_model, get_integrated_gradients
import matplotlib.pyplot as plt
from io import BytesIO

# Load the model
model = load_model()

# Load MNIST dataset for visualization
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Streamlit interface
st.title("Digit Attribution Visualization")

# Sidebar selection
selected_digit = st.sidebar.selectbox("Select a digit to view", list(range(10)))
page_number = st.sidebar.number_input("Page Number", min_value=0, step=1, value=0)

# Initialize session state variables
if 'selected_image_idx' not in st.session_state:
    st.session_state['selected_image_idx'] = None
if 'attributions' not in st.session_state:
    st.session_state['attributions'] = None  # Initialize to store the attributions

# Filter dataset by selected digit
digit_images = [data for data in test_dataset if data[1] == selected_digit]
total_images = len(digit_images)
images_per_page = 16  # 4x4 grid
start_idx = page_number * images_per_page
end_idx = min(start_idx + images_per_page, total_images)

# Display a grid of digit images
st.subheader(f"Images of digit '{selected_digit}'")
cols_per_row = 4
rows = 4

# Loop to display images in a grid
for i in range(rows):
    col_items = st.columns(cols_per_row)
    for j, col in enumerate(col_items):
        idx = start_idx + (i * cols_per_row + j)
        if idx < end_idx:
            img, label = digit_images[idx]
            img_np = img.squeeze().numpy()
            if col.button(f"Select Image {idx}", key=idx):
                st.session_state['selected_image_idx'] = idx  # Store selected image in session state
            col.image(img_np, width=80, caption=f"ID: {idx}")

# Display the "Test" button and handle the selected image
if st.session_state['selected_image_idx'] is not None:
    st.write(f"Selected Image ID: {st.session_state['selected_image_idx']}")
    test_image, test_label = digit_images[st.session_state['selected_image_idx']]

    # Display test button
    if st.button("Test"):
        # Process selected image through the model for integrated gradients
        test_image = test_image.unsqueeze(0)  # Add batch dimension
        attributions = get_integrated_gradients(model, test_image, test_label)
        st.session_state['attributions'] = attributions  # Store attributions in session state

# Display the heatmap only if attributions have been calculated
if st.session_state['attributions'] is not None:
    fig, ax = plt.subplots()
    im = ax.imshow(st.session_state['attributions'], cmap='hot', interpolation='nearest')
    ax.set_title(f"Integrated Gradients for label: {test_label}")
    fig.colorbar(im, ax=ax, orientation='vertical')

    # Convert plot to PNG for Streamlit display
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)  # Display the heatmap image in Streamlit
    plt.close(fig)
