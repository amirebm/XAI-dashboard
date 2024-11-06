import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import SimpleCNN, load_model, get_integrated_gradients # Import from model.py

# Function to load the saved model
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("simple_cnn.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

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

# Filter dataset by selected digit
digit_images = [data for data in test_dataset if data[1] == selected_digit]
total_images = len(digit_images)
images_per_page = 16  # 4x4 grid
start_idx = page_number * images_per_page
end_idx = min(start_idx + images_per_page, total_images)

# Display a grid of digit images
st.subheader(f"Images of digit '{selected_digit}'")
cols_per_row = 4  # Define how many columns per row
rows = 4
selected_image_idx = None

# Loop to display images in a grid
for i in range(rows):
    col_items = st.columns(cols_per_row)  # Create a list of column objects
    for j, col in enumerate(col_items):
        idx = start_idx + (i * cols_per_row + j)
        if idx < end_idx:
            img, label = digit_images[idx]
            img_np = img.squeeze().numpy()
            if col.button(f"Select Image {idx}", key=idx):
                selected_image_idx = idx
            col.image(img_np, width=80, caption=f"ID: {idx}")


# If an image is selected
if selected_image_idx is not None:
    st.write(f"Selected Image ID: {selected_image_idx}")
    test_image, test_label = digit_images[selected_image_idx]

    # Display test button
    if st.button("Test"):
        # Process selected image through the model for integrated gradients
        test_image = test_image.unsqueeze(0)  # Add batch dimension
        attributions = get_integrated_gradients(model, test_image, test_label)

        # Show the attribution map using matplotlib
        fig, ax = plt.subplots()
        ax.imshow(attributions, cmap='hot', interpolation='nearest')
        ax.set_title(f"Integrated Gradients for label: {test_label}")
        plt.colorbar(ax=ax, orientation='vertical')

        # Convert plot to PNG for Streamlit display
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
        plt.close(fig)