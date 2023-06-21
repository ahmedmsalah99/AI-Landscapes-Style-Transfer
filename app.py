import streamlit as st
import tkinter as tk
from tkinter import filedialog
import tempfile
import os,sys
new_directory = os.path.abspath("QuantArt")
sys.path.insert(0, new_directory)
from QuantArt.generate_art import generate_landscape_art
app_state = st.experimental_get_query_params()  

ckpt_path = 'QuantArt/logs/landscape2art/checkpoints/last.ckpt'
fp = tempfile.TemporaryFile()
result_dir = None
# UI
# Set up tkinter
root = tk.Tk()
root.withdraw()
# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)
st.title('LandScape Art Generator')
images_files = st.file_uploader("Please choose your images",type=['jpg','jpeg','png'],accept_multiple_files=True)
styles_files = st.file_uploader("Please choose your styles",type=['jpg','jpeg','png'],accept_multiple_files=True)
clicked = st.button('Results Folder')

images_paths = []
for file in images_files:
    file_path = os.path.join("tempDir/", file.name)
    with open(file_path,"wb") as f: 
        f.write(file.getbuffer())             
    images_paths.append(file_path)
styles_paths = []
for file in styles_files:
    file_path = os.path.join("tempDir/", file.name)
    with open(file_path,"wb") as f: 
        f.write(file.getbuffer()) 
    styles_paths.append(file_path)

if clicked:
    result_dir = filedialog.askdirectory(master=root)
    dirname = st.text_input('Selected folder:', result_dir)
    st.experimental_set_query_params(result_dir=result_dir)
if app_state.get('result_dir') is not None:
    result_dir = app_state.get('result_dir')[0]
    dirname = st.text_input('Selected folder:', result_dir)
    
    

    
if st.button("Generate Art!!"): 
    if result_dir is not None and len(images_paths)>0 and len(styles_paths)>0:
        generate_landscape_art(images_paths,styles_paths,ckpt_path,result_dir)
        st.success("Done")
    else:
        st.text("Fill the fields first")