from imageio import mimsave, imread
import os
import glob

SKIP_COUNT = 50
FPS = 3
MODEL_NAME =  'DCGAN'

def generate_gif(model_name, skip_count, fps):
    path = os.path.join('../models', model_name, 'generated_images')
    img_files = sorted(glob.glob(os.path.join(path, '*.jpg')), key=os.path.getmtime)

    images = []
    i = 0
    for img in img_files:
        if i == 0:
            images.append(imread(img))
        elif i >= skip_count:
            images.append(imread(img))
            i = 0
            continue

        i += 1

    print("images reading complete...")
    print("creating %s gif with fps: %d..." % (model_name, FPS))
    mimsave(os.path.join(path, 'samples_%s.gif' % model_name), images, fps=fps)



if __name__ == "__main__":
    generate_gif(MODEL_NAME, SKIP_COUNT, FPS)