## Neural Style Transfer

#### Input:
- Initial Image to change 
- Style Image
- Output image (initially just noise or the initial image)

Using VGG19 ( a pretrained model )

#### Output:

- Output image


## How to run :

Run the `main.py` script with specified paths for the content and style images. Additional optional arguments allow customization of the training and generation process.

```bash
python main.py --content_image "images/building.jpg" --style_image "images/van_gogh.jpg" --output_dir "output_images" --total_steps 3000 --alpha 1 --beta 0.02
```

## Example :

<img src="https://github.com/tarekakrout2021/NeuralStyleTransfer/blob/main/example.png" width=80% height=80%>