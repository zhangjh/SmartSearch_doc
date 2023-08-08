# Developed an APP using CLIP for retrieving images based on text input

This year can be considered a groundbreaking year for AI. Since the beginning of the year, ChatGPT has gained popularity, and numerous AI models have emerged one after another. Alongside these models, various applications based on large models have appeared like bamboo shoots after a spring rain.

I have been following this field since ChatGPT's launch in November last year. From the initial awe of GPT to conducting in-depth research and attempting to utilize it, I have developed small programs and some derivative applications using the ChatGPT API during this period, such as a Chrome Translate plugin, text-to-speech conversion, document summarization, and more. In my work, I have also employed ChatGPT's capabilities to create vertical knowledge base applications, Q&A robots, and even a side project for beautifying QR codes with Stable Diffusion.

However, apart from ChatGPT significantly improving the convenience of knowledge retrieval (provided you can accurately recognize its nonsensical content) and enhancing productivity, there doesn't seem to be much practical value in other peripheral products at the moment.

When I was researching other products of OpenAI, I discovered CLIP, which they released in 2021. It is a model that can vectorize text or images, and I instantly had a new idea: ChatGpt can use natural language for conversation, so can I use CLIP to search for images with natural language? If so, I feel this product would be more practical than the current crop of products that simply repackage ChatGpt.

## What is CLIP?
CLIP (Contrastive Language-Image Pre-Training) is a neural network model developed by OpenAI based on image-text pair training. It can predict the most relevant text snippet for a given image without directly optimizing the task through natural language instructions, similar to GPT's zero-shot capability.

It vectorizes and extracts features from both text and images, mapping them to the same high-dimensional space, allowing comparisons of text and image similarity by comparing vector distances.

Here is an official schematic diagram of the principle:
![CLIP.png](https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F80eed3bc-aed2-4711-8ce7-0f70699dd092%2FCLIP.png?table=block&id=88419832-ce30-4d6c-a429-df896ba4f5fd&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=2000&userId=&cache=v2)
The principle is straightforward, a typical dual-tower model.

First, a model is pre-trained through a large number of text-image pairs, which mainly encodes text and images, mapping them to the same high-dimensional space, resulting in a series of vector features.

With the pre-trained model, during using, the user's input text is encoded by the model encoder to obtain a text vector. Then, using vector retrieval algorithms, the most similar image vector is retrieved by calculating the distances between different vectors, thus obtaining the image with the highest relevance to the text.

## Can it be ported to a mobile phone?
To port CLIP to a mobile phone and support Chinese text, I used the [open-source Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) model and converted the model format to ONNX format, allowing it to run on low-performance, GPU-less devices like mobile phones.
According to the pre-trained model data provided by Chinese-CLIP, considering the overall effect and model size, I chose to use the ViT-B/16 model and converted it to an ONNX model. After conversion, the size of the image and text models was close to 1 GB, which is a massive presence on the device. Unlike other types of apps, AI models need to be loaded into memory in their entirety, and this large model is likely beyond the capacity of most mobile phone RAM. I need to find a way to reduce the model size.
Since I am using Chinese-CLIP for its Chinese support function, the image model can use the official CLIP directly. Therefore, I used the CLIP to export the encoder of the image model and removed other unnecessary functions:

```python
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
device = "cpu"

model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
dummy_image = torch.randn(1, 3, 224, 224)  

# Make sure the model is in evaluation mode  
# Convert to ONNX  
torch.onnx.export(model.visual, dummy_image, "image_encoder.onnx", opset_version=12,  
input_names=["image"], output_names=["embOutput"],  
operator_export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)  
```
For the text model, I used the fp32 model converted from Chinese-CLIP. The original model size is still considerable, so I performed a quantization operation on the model:  
```python
import onnx  
from onnxruntime.quantization import quantize_dynamic, QuantType  
model_fp32 = "vit-b-16.txt.fp32.onnx"  
model_quant = "txt_quant2.onnx"  
quantized_model = quantize_dynamic(model_fp32, model_quant)  
img_model_fp32 = "img_encoder.onnx"  
img_model_quant = "img_encoder_quant.onnx"  
img_quantized_model = quantize_dynamic(img_model_fp32, img_model_quant)  
```
After quantization, the text model size is 197MB, and the image model size is 84MB, which is an acceptable size for mobile devices.  
By separating the text and image models, we can ensure that both models are not loaded simultaneously, only loading the image model when needed and the text model when needed, further reducing the requirements for mobile phone RAM.  
The life cycle of the two models in actual use is as follows:  
![SmartSearch.png](https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F5fac1bac-9e75-4733-8bd9-233ea7621574%2FSmartSearch.png?table=block&id=1c571103-dfe3-4973-bab4-ffacab3a5588&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=2000&userId=&cache=v2)
During the image index construction phase, only the image model is used, and it is not needed once the construction is complete. The text retrieval phase only uses the text model.  

## How to solve privacy and security concerns?  
To make an app that needs to scan photo albums, privacy and security issues must be a core focus.  
As described above, both the construction of the index and text retrieval are based on local model calculations, and the application itself is entirely independent of cloud services. This is why it is necessary to go through the trouble of porting the model to the mobile device.  
CLIP itself is the basis of many text-to-image models, such as Stable Diffusion or OpenAI's vector embedding calculations. To dispel users' concerns about privacy and security, the model is ported to mobile devices and runs locally without relying on cloud services.  

## Usage demonstration 
Before developing this app, I often struggled to find a picture in my memory, having to flip through my photo album for a long time, and sometimes giving up because I didn't want to browse the album.  
With the help of CLIP, digging through a vast phone album is no longer a headache.  
With over 2000 photos in my phone album, it takes just seconds to search for a picture. I can often find long-forgotten memories and evoke emotional ripples in my heart with a single thought.  
Here, I demonstrate the usage of the smart search with Unsplash's photo library as an example:  
1. Brief keyword description
   
   <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fe132f0df-63b6-4d73-8cce-37c44a915468%2Fss4.jpg?table=block&id=7ef5cdfd-8b0a-4bd8-bebe-8fd8504d0b13&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=2000&userId=&cache=v2" width="192px" />
   
   
   <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fef222ea8-fb1e-433f-9e56-e896198837ad%2Fss3.jpg?table=block&id=dbaa0dce-8c80-49cd-9e55-4452187fddaa&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=380&userId=&cache=v2" width="192px" />


   <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F997f98df-e291-4ca5-969d-1ab1b4973b72%2Fss5.jpg?table=block&id=6f8269e4-32bd-481a-b69a-82b264913b5a&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=380&userId=&cache=v2" width="192px" />
   
2. Try a longer sentence
   
    beautiful night sky has stars
  
    <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fef3897a3-080e-4de8-9398-43732251e4f9%2F20230807174336.jpg?table=block&id=c9dddba2-f1f2-4019-bb27-b7e3c2db2c06&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=380&userId=&cache=v2" width="192px" />
  
    winding mountain road, has lush green trees
  
    <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F821e0792-7319-4020-bd58-81b8cc89a0e7%2F20230807174546.jpg?table=block&id=84fca10f-3ba0-4a98-bc6c-a5bdb107c208&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=380&userId=&cache=v2" width="192px" />
  
3. Try a few more photos from my album
   
     <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fd0b17957-cf4e-4dfb-9b34-dacf99215b1f%2F20230807175234.jpg?table=block&id=fcd38f4b-d62f-4a85-8b30-77d996ed42e6&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=380&userId=&cache=v2" width="192px" /> 

   Amazing, even text ‘kobe bryant’ can be found..

4. Certificates should be a high-frequency query scenario, and it's not a problem
   
    <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F29046a6c-dff8-4a74-9560-07f4c001c3c1%2Fss1.jpg?table=block&id=e24737ee-6ed1-4f67-aade-ade305e79685&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=380&userId=&cache=v2" width="192px" /> 
  
5. Even the emotions conveyed by the pictures can be found
   
    <img src="https://zhangjh.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fdbf3120c-8482-49f8-8f1a-2825b70a927b%2Fss2.jpg?table=block&id=e59f201b-3a2e-4322-927b-b0d0031f879f&spaceId=af444414-65d9-4b69-bb58-adb10a51ef3c&width=380&userId=&cache=v2" width="192px" />

To comply with Google Play's requirement that the installation package not exceed 150MB in size (otherwise, a special resource distribution is required), the model files need to be downloaded during the first use. The total size of the text and image model files is less than 300MB, and the download may take a few minutes.  
On initial run or after self-building when new images are added, the app will prompt for index construction, which requires photo album permissions. Of course, you may wonder if the app is truly running locally. In that case, you can wait for the model to download and grant photo album permissions to start building the index.  
The process of building the index only needs to be performed once during the initial run. Subsequently, incremental building will be done when new images are added.

## Other Information
1. App official website:

    https://ss.zhangjh.me/index_en.html
2. Device requirements:

    Android 10+
3. Download channel:

    [Google Play](https://play.google.com/store/apps/details?id=me.zhangjh.smart.search.en)
4. Contact information:
   
    Email: [zhangjh_initial@126.com](mailto:zhangjh_initial@126.com)

    Twitter: [@Dante_Chaser](https://twitter.com/dante_chaser)
