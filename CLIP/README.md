# CLIP (Contrastive Language-Image Pre-training) Implementation

This implements Contrastive Language-Image Pre-training (CLIP), trained on the Flickr30k dataset and evaluated through ImageNet-1K Zero-shot Classification.

## Project Overview

- **Objective:** Learning visual concepts from natural language supervision by aligning image and text embeddings in a shared latent space.
- **Methodology:** Utilizing Weight Transplantation from pretrained ResNet50 and CLIP-Text models to boost learning efficiency and performance.
- **Key Success:** Achieved significant zero-shot transfer capabilities on unseen datsets (ImageNet-1K).

## Architecture

The model consists of two separate encoders mapped to a joint multimodal embedding space via learnable projection heads.

- **Image Encoder:** ResNet50 (Pretrained on ImageNet).
- **Text Encoder:** Transformer-based encoder (Pretrained CLIP-Text).
- **Projection Heads:** Two separate linear layers ($2048 \to 512$) for image and ($512 \to 512$) for text features, followed by L2 normalization.
- **Similarity Metric:** Scaled cosine similarity with a learnable temperature parameter $\tau$.

## Experimental Results

### Flickr30k Zero-shot Transfer  

![Flickr30k_zero_shot_transfer](outputs/flickr-30k_zero_shot_transfer.png)  

| Metric | Result |
| :---: | :---: |
| Top-1 Accuracy | 42.05% |
| Top-5 Accuracy | 82.11% |  

### ImageNet-1K Zero-shot Transfer  

Evaluated on the ImageNet validation set (50,000 images) without any fine-tuning on ImageNet labels.  

![ImageNet-1k_zero_shot_transfer](outputs/ImageNet-1k_zero_shot_transfer.png)  

| Metric | Result |
| :---: | :---: |
| Top-1 Accuracy | 11.88% |
| Top-5 Accuracy | 26.42% |

>  For 1000 classes, the model performs **118x better than random guess (0.1%)**, demonstrating successful zero-shot knowledge transfer.

### Flickr30k Retrieval Task  

Evaluating the model's ability to find corresponding pairs in the Flickr30k test set.

![Flickr-30k_retrieval](outputs/flickr-30k_retrieval.png)

| Task | Recall@1 | Recall@5 | Recall@10 |
| :---: | :---: | :---: | :---: |
| Image-to-Text | 29.98% | 58.13% | 70.18% |

## Training Progress

The model was trained for 30 epochs using a contrastive objective. The loss decreased steadily, showing rapid convergence within the first 10 epochs.

![CLIP_Loss](outputs/CLIP_Loss.png)

### Hyperparameters

| Parameter | Value |
| :---: | :---: |
| Optimizer | AdamW ($lr = 2 \times 10^{-5}$, weight decay $=0.01$) |
| Batch Size | 96 |
| Scheduler | Cosine Annealing with Warmup |
| Temperature ($t$) | Learned (Initial $\tau \approx 0.07$) |
| Max Epochs | 30 |

### Result

Here is a real example of the model performing an Image-to-Text search from the Flickr30k test set.  
The model successfully identifies the key subjects and actions in the query image.

| Query Image | Top 5 Predicted Captions |
| :---: | :--- |
| ![Query Image](outputs/media_images_retrieval_query_image_0_c45886724bc68f2c432e.png) | 1. **(Matched)** A person in a blue and white baseball uniform is standing on one foot while holding a bat . <br> 2. Man playing for a baseball team with a blue protective hat and baseball uniform , playing for South Carolina , is swinging the bat and missing the ball . <br> 3. The guy is in a baseball uniform , in a baseball field , and is throwing a baseball <br> 4. A baseball pitcher on the mound in a black , green and white uniform prepares to release his pitch . <br> 5. A baseball pitcher wearing a purple and white uniform throwing a baseball from the mound . |
