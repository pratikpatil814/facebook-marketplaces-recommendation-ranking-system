import os
from PIL import Image

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = r'C:\Users\prati\Documents\facebook-marketplaces-recommendation-ranking-system\data\training_data\images_fb\images'
    dirs = os.listdir(path)
    final_size = 512
    for n, item in enumerate(dirs, 1):
        im = Image.open(path +r'/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(fr'C:\Users\prati\Documents\facebook-marketplaces-recommendation-ranking-system\data\training_data\new_images_fb\{n}_resized.jpg')
