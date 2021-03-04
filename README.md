# Malivar ML Engineer Test

There are several ways of generating of people who doens't exists. The most common thing is to use GAN model. 
To have a control under what is going on generation process we need to work with latent features. Unfortunatelly, they are not as intepriable as it can be wished, so there are several approaches of working with them.
Another way to work with photo is to convert it to 3D space and do manipulations with post and emotions there.

During doing this test I explore several approaches and provide here a brif description of each with paper and code.

### Approach 1. Supervised Learning.
In this case we will use Streamlit tool for visualizing Machine Learning projects.
For launching the solution (you need Anaconda):
```sh
conda create -n python3.6 python=3.6
conda activate python3.6
cd demo-face-gan
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This approach is supervised and uses 40 features, that are labeled manually before training.
[<img src="https://miro.medium.com/max/1400/1*l6ug8cLOpd_TtxxgWlnw1Q.png">](https://blog.insightdatascience.com/generating-custom-photo-realistic-faces-using-ai-d170b1b59255)
Truly, this approach doesn't show amazing results and need time to play with params and achieve good looking and realistic photo. Also, this approach doesn't take into account pose changing.

### Approach 2. Unsupervised Learning.
Another way of generating face with params is unsupervised approach. The approach is fully described in [this paper](https://arxiv.org/pdf/2004.02546.pdf).
The main idea briefly: author apply PCA on GAN latent space and find out how to connect the directions in projections with the changes on image.

To explore this aproach run in Google Colab **GANSpace.ipynb**.

<img src="https://i.ibb.co/yBnh8v9/Screenshot-2021-03-04-at-17-01-56.png" alt="Screenshot-2021-03-04-at-17-01-56">

On the image above authors shows how to play with different PCA components and what part of face will change. I need to mentione here, that I tried to repeat the parameters from the paper, but my success attempt was only to find how to deal with rotation of the face on the photo. 
This approach doesn't deal with vertical pose changing.

### Approach 3. Playing with StyleGAN2.

Let's condider the main idea of this approach on the practical case. Let say, we need to obtain photos of people with glasses. StyleGAN is a good choice for this.
The algorithm for this approach is the following:
* Generate N = 1000 random faces. 
* Select those ones, which have glasses.
* Extract latent vectors of random two photos from selected dataset, got the point somewhere in the middle of these two vector and put it to generator (see the picture).

<a href="https://ibb.co/FsffQ2z"><img src="https://i.ibb.co/Q8qqWLM/Screenshot-2021-03-04-at-17-20-29.png" alt="Screenshot-2021-03-04-at-17-20-29" border="0"></a>

The notebook for this approach is **gan_explore.ipynb**.

This approach has limitations such as missing the possibility to tune the algorithm parameters. But the advantage is the good quality of generated photos (thanks to StyleGAN2 model).

### Approach 4. StyleGAN with 3DMM.

The original paper for this approach: https://arxiv.org/pdf/2004.11660.pdf
The key advantages of this approach is the working not only with 2D photo, but also incorporating 3D.
The algorithm is as following: learn 3D represenation of the provided photo then make manipulations with mesh structure and finally render the result.
The known problem of this approach is the imperfect rendering after collectin the mesh structure of the photo. This is the active area of research now.

There is code exmples provided with paper, but the problem is that authors provide only random image generation. I store it at **DiscoGAN.ipynb** notebook.
To recreate the results from the original paper and to have a possibilities to work with any photo, some additional manupulations needed.

I didin't do it in this ML Engineer Test due to lack of time, but I have a vision how it can be done (based on discussions on paper code):
* Convert the image into the latent space. There is an encoder in the original paper, but also [another one](https://github.com/pbaylies/stylegan-encoder), which was stated to be good at achieving good looking results.
* Convert the encoded image to 3D space and make the manipulations.
* Render the resulting images.

Here I play with encoder: 
* **StyleGAN_Encoder.ipynb** for encode any image with face and check the decoding result.
*  **Play_with_latent_directions.ipynb** for playing with endoded images params: emotions, age, gender.

Unfortunatelly, I didn't connect two solutions :(

### To sum up
The most powerful approach is the 4th one, because it incorporates both 2D and 3D dimensions. 

### Resources, that I also study for this task
I begin to tackle the problem from 3D point, so my starting point was to explore current methods of working with 3D data:

[3D Morphable Face Models - Past, Present and Future](https://arxiv.org/pdf/1909.01815.pdf)
[StyleGAN original paper](https://arxiv.org/pdf/1812.04948.pdf)
First 2 Sections from [Face Image Analysis using a Multiple Features Fitting Strategy](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.471.3366&rep=rep1&type=pdf)
My [starting point](https://github.com/YadiraF/face3d) with example of working with mesh structures. My first idea was to use this code for image manipulation, but then I figure out, that obtaining the good 3DMM of the face is not as easy, as I expected. I look into these two ones: https://github.com/microsoft/Deep3DFaceReconstruction, https://github.com/tranluan/Nonlinear_Face_3DMM, but failed in launching it on my MacBook or Colab.
