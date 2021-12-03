#!/usr/bin/python3

# Import modules for CGI handling 
import cgi,sys,os
import cgitb
cgitb.enable()

sys.path.append('/usr/lib/python3/dist-packages/')
sys.path.insert(0, os.getcwd())
from tflite_runtime.interpreter import Interpreter
import numpy as np
import argparse
from PIL import Image

print("Content-type:text/html\r\n\r\n")
print("<html>")
print("<head>")
print("<title>Mobilenet V1</title>")
print("</head>")
print("<h1>Senior project Mobilenet V1 model</h1>") 
print("<form action=\"Mobilenet_CGI.py\" method=\"post\"enctype=\"multipart/form-data\">")
print("<input type=\"file\" name=\"file1\">")
print("<input type=\"submit\">")
print("</form>")

form = cgi.FieldStorage()

fn = "Mobilenet/Mobilenet_Images/TigerShark_1.jpeg"

if 'file1' in form:
    fileitem = form['file1']
    fn = os.path.join('./Mobilenet', os.path.basename(fileitem.filename))
    open(fn, 'wb').write(fileitem.file.read())
    print("<div>The file",fn," was uploaded</div>")
else:
    print("<div> No file was uploaded</div>")

print("<br><br>")

filename = fn
model_path = "Mobilenet/Model_Files/mobilenet_v1_1.0_224_quant.tflite"
label_path = "Mobilenet/Model_Files/labels_mobilenet_quant_v1_224.txt"
top_k_results = 3

with open(label_path, 'r') as f:
    labels = list(map(str.strip, f.readlines()))

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read image
img = Image.open(filename).convert('RGB')

# Get input size
input_shape = input_details[0]['shape']
size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

# Preprocess image
img = img.resize(size)
img = np.array(img)

# Add a batch dimension
input_data = np.expand_dims(img, axis=0)

# Point the data to be used for testing and run the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtain results and map them to the classes
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# Get indices of the top k results
top_k_indices = np.argsort(predictions)[::-1][:top_k_results]

print("<body style=\"background-color:#5A9BB0;\">")
print("<h2><u>Model Predictions:</u></h2>")
print("<p>",labels[top_k_indices[0]],"(",round((predictions[top_k_indices[0]] / 255.0)*100,2),"%)","</p>")
print("<p>",labels[top_k_indices[1]],"(",round((predictions[top_k_indices[1]] / 255.0)*100,2),"%)","</p>")
print("<p>",labels[top_k_indices[2]],"(",round((predictions[top_k_indices[2]] / 255.0)*100,2),"%)","</p>")
print("<br><br>")
print("<p>Note: Default Image is TigerShark_1.jpeg</p>")
print("</body>")
print("</html>")
exit()
