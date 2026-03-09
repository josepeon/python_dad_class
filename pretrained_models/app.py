#pip install streamlit
import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch
from torchvision.transforms import ToTensor

# Load the ResNet50 model
weights = ResNet50_Weights.DEFAULT
transforms = weights.transforms()
model = resnet50(weights=weights)

# Set up the Streamlit app
st.header("Pretrained Model")
st.write("This is a placeholder for the pretrained model application.")
image = st.file_uploader("Upload a file to process")
if image is not None:
    st.write("File uploaded successfully!")
    st.write(image.name)
st.image(image)
#Convert image to tensor and remove alpha channel
image = Image.open(image)
t = ToTensor()
image = t(image)
transformed_image = transforms(image[:-1, :, :])
# Apply the transforms and make prediction
prediction = model(transformed_image.unsqueeze(0))
pred_val = torch.argmax(prediction)
st.write(pred_val)
st.write(weights.meta["categories"][pred_val])

#Jacob's code below
###
#pip install streamlit
import streamlit as st 
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import decode_image
from PIL import Image
import torch
from torchvision.transforms import ToTensor
#bring in model and all
weights = ResNet50_Weights.DEFAULT
transforms = weights.transforms()
model = resnet50(weights = weights)
#make our app with a header and file upload
st.header('My Model')
st.write('This is my neural net.')
image = st.file_uploader('Upload A Picture')
st.image(image)
#convert image to right format
image = Image.open(image)
t = ToTensor()
image = t(image)
transformed_image = transforms(image[:-1, :, :])
#pass it through the model
prediction = model(transformed_image.unsqueeze(0))
pred_val = torch.argmax(prediction)
st.write(pred_val)
st.write(weights.meta['categories'][pred_val])
###