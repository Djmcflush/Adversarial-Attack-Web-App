import streamlit as st
from PIL import Image,ImageFilter
import torchvision.transforms as transforms
from torchvision import *
from torch import *
import time
from modelclass import *
from LabelToText import *
import random
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import os
import gdown
from pathlib import Path
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
def Adversarial(image, y_true, model):
    epsilons = [0.03, .05]
    examples = []
    xgrads = []
    image = image.resize((224,224))
    image = transform(image).view(1,3,224,224)
    img_variable = Variable(image, requires_grad= True)
    output = model.forward(img_variable)
    target = Variable(torch.LongTensor([y_true]),requires_grad=False)
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, target)
    loss_cal.backward(retain_graph=True)
    
    for eps in epsilons:
        x_grad = torch.sign(img_variable.grad.data)          #calculate the sign of gradient of the loss func (with respect to input X) (adv)
        x_adversarial = img_variable.data + eps * x_grad          #find adv example using formula shown above
        #output_adv = model.forward(Variable(x_adversarial))   #perform a forward pass on adv example
        #x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0]]    #classify the adv example
        #op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
        #adv_pred_prob =  round((torch.max(op_adv_probs.data, 1)[0][0]) * 100, 4)      #find probability (confidence) of a predicted class
        examples.append(x_adversarial)
        xgrads.append(x_grad)
    rand_idx = random.randint(0, len(examples)-1)
    random_img = examples[rand_idx]
    random_x_grad = xgrads[rand_idx]
    return random_img, random_x_grad

@st.cache
def AdverarialTraining():
    #load in adverssarial network
    base = os.getcwd()
    save_dest = os.path.join(base, 'model')
    if os.path.exists(save_dest):
        pass
    else:
        os.mkdir(save_dest) 
    f_checkpoint = os.path.join(save_dest,"resnet50.pt")
    
    if not os.path.exists(f_checkpoint):
        url = "https://drive.google.com/file/d/1YYNy3djfxl3hHaFARsSUqhjnwxliA5Gv/"
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdown.download(url, f_checkpoint, quiet=False)
    print(os.listdir(save_dest))
    
    model = torch.load(f_checkpoint, map_location=device)
    model.eval()
    return model   
    
def postprocess(x_grad, x_adv):
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.moveaxis(x_grad, 0,2)
    x_grad = np.clip(x_grad, 0, 1)

    x_adv = x_adv.squeeze(0)
    x_adv = x_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    #x_adv = np.moveaxis(x_grad, 0,1)
    #x_adv = np.clip(x_adv, 0, 1)
    return x_adv, x_grad

st.write('''<style>
            body{
            text-align:center;
            background-color:#FFFFFF;

            }

            </style>''', unsafe_allow_html=True)




st.title('Adversarial Attacks On Resnet 50!')


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


#loading the model
resnet = Resnet()

#loaded_densenet169 = Densenet169()
#loaded_densenet169.load_state_dict(torch.load('densenet169.pt',map_location=torch.device('cpu')))
resnet.eval()

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.001)

'Resnet50 loaded!'
st.text('Resnet50 loaded')

col1, col2, col3 = st.beta_columns(3)
col1.header("Original")
col2.header("Peturbation")
col3.header("Adversarial")


file_type = 'jpg'


uploaded_file = st.file_uploader("Choose an Image!",type = file_type)


if uploaded_file != None:

    image = Image.open(uploaded_file)
    image = image.filter(ImageFilter.MedianFilter)
    im = image
    col1.image(image, caption='Uploaded Image.', use_column_width=True)
    #col1.write(image.size)
    origial_size = image.size
    image = image.resize((224,224))
    image = transform(image).view(1,3,224,224)
    pred  = resnet.forward(image)
    proba,idx = torch.max(torch.sigmoid(pred),dim = 1)

    proba = proba.detach().numpy()[0]
    idx = idx.numpy()[0]    



    label_oracle = Labels()
    pred_label = label_oracle.label(idx)
    label_str = pred_label.split(',')
    for x in label_str:
        col1.text(x)
    col1.write('confidence {:0.3f}'.format(float(proba)))


    
    
    
    im, x_grad =  Adversarial(im, idx, resnet)
    
    
    #im = im.numpy()
    #im = im.squeeze()
    #x_grad = x_grad.numpy()
    #x_grad = x_grad.squeeze()
    
    #x_grad = x_grad.resize((224,224))
    #x_grad = np.moveaxis(x_grad, 0, 2)
    im, x_grad  = postprocess(x_grad, im)

    #col2.write(x_grad.shape)
    #x_grad = transform(x_grad).view(1,3,224,224)

    
    col2.image(x_grad, caption='Invisible Perturbation you added to the image', use_column_width = True)
    first_idx = idx
    
        
    
    temp = cv2.resize(im, dsize=origial_size, interpolation=cv2.INTER_CUBIC)
    temp = np.clip(temp, 0, 1)
    col3.image(temp, caption='Adversarial Image.', use_column_width=True)
    im = Image.fromarray(np.uint8(im)*255)
    im = im.resize((224,224))
    im = transform(im).view(1,3,224,224)


    pred  = resnet.forward(im)
    proba,idx = torch.max(torch.sigmoid(pred),dim = 1)
    
    proba = proba.detach().numpy()[0]
    idx = idx.numpy()[0]
    second_idx = idx
    if first_idx == second_idx :
        col2.write("Wait... How'd You do that? *Double or nothin* :sunglasses: ")
    else:
        col2.text("Congrats!")
        col2.text("Succesful Attack!")
        col2.write(":fire:")
    pred_label = label_oracle.label(idx)
    label_str = pred_label.split(',')
    for x in label_str:
        col3.text(x)
    col3.write('confidence {:0.3f}'.format(float(proba)))
    st.subheader("Explanation Below")
    
    
    with st.beta_container():
        st.write('''<style>
            body{
            text-align:justified;
            background-color:#FFFFFF;

            }

            </style>''', unsafe_allow_html=True)
        for col in st.beta_columns(1):
            col.write("Okay so your wondering how this is all working?")
            col.latex(r'''
            adversarial_{image} =  \sin(\triangledown_{input} J(0,input,y_{true}))
             ''')
            col.write("""The attack you just implemented is known as the FGSM attack.
             This attack first introduced by Ian Goodfellow uses the optimized gradients
             of the nerual networks output (the processed image tensor) to maximize loss
             between the target class and the predicted class. The attack optimizes on 
             the L_infity metric between the input image and the adversarial image. 
             The L-Infinity distance judges the pixel with the largest amount of change. 
             This attack takes "steps" (Cough Cough Gradient Descent Sound familiar?) 
             to figure out which direction creates the most loss. In this way were not 
             actually targeting any particular label but instead ensuring that the model 
             doesnt predict the true label. """)
            col.write("https://www.mdpi.com/2079-9292/9/8/1284/htm")
    with st.beta_container():
        for col in st.beta_columns(1):
            col.write("Results After Adversarial Training")
            adv_model = AdverarialTraining()
            pred  = adv_model.forward(im)
            proba,idx = torch.max(torch.sigmoid(pred),dim = 1)
            proba = proba.detach().numpy()[0]
            idx = idx.numpy()[0]
            pred_label = label_oracle.label(idx)
            label_str = pred_label.split(',')
            for x in label_str:
                col.text(x)
            col.write('confidence {:0.3f}'.format(float(proba)))
            col.image()
            col.image(temp, caption='After Adversarial training', use_column_width=True)
