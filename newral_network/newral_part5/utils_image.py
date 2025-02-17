import os 
import random 
import shutil
import piexif

def train_test_split(src_folder, train_size = 0.8):
     #what is shutil here 
    shutil.rmtree(os.path.join(src_folder, "Train", "Cat"), ignore_errors = True)
    shutil.rmtree(os.path.join(src_folder, "Train", "Dog"), ignore_errors = True)
    shutil.rmtree(os.path.join(src_folder, "Test", "Cat"), ignore_errors = True)
    shutil.rmtree(os.path.join(src_folder, "Test", "Dog"), ignore_errors = True)

    #what is makedirs
    #what is OS exactly
    os.makedirs(os.path.join(src_folder, "Train", "Cat"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "Train", "Dog"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "Test", "Cat"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "Test", "Dog"), exist_ok=True)

    #what do you mean by _,_, cat_images

    src_folder = 'C:\Real junkies\sem 8\Datasets\cats_and_dogs\PetImages'
    
    _,_, cat_images = next(os.walk(os.path.join(src_folder, "Cat")))
    files_to_be_removed = ['Thumbs.db', '666.jpg', '835.jpg']
    for file in files_to_be_removed:
        cat_images.remove(file)  #why the files are being removed
    
    num_cat_images = len (cat_images)
    num_cat_images_train = int(train_size * num_cat_images)
    num_cat_images_test = num_cat_images - num_cat_images_train

    #what is next function here and what is os.walk
    _,_, dog_images = next(os.walk(os.path.join(src_folder, "Dog")))
    files_to_be_removed = ['Thumbs.db', '11702.jpg']
    for file in files_to_be_removed:
        dog_images.remove(file)

    num_dog_images = len(dog_images)
    num_dog_images_train = int(train_size * num_dog_images)
    num_dog_images_test = num_dog_images - num_dog_images_train


    cat_train_images = random.sample(cat_images, num_cat_images_train)
    for img in cat_train_images:
        shutil.copy (src = os.path.join(src_folder, "Cat", img), dst = os.path.join(src_folder, "Train", "Cat", img))
    
    cat_test_images = [img for img in cat_images if img not in cat_train_images]
    for img in cat_test_images:
        shutil.copy (src = os.path.join(src_folder, "Cat", img), dst = os.path.join(src_folder, "Test", "Cat", img))
    

    
    dog_train_images = random.sample(dog_images, num_dog_images_train)
    for img in dog_train_images:
        shutil.copy (src = os.path.join(src_folder, "Dog", img), dst = os.path.join(src_folder, "Train", "Dog", img))
    
    
    dog_test_images = [img for img in dog_images if img not in dog_train_images]
    for img in dog_test_images:
        shutil.copy (src = os.path.join(src_folder, "Dog", img), dst = os.path.join(src_folder, "Test", "Dog", img))
    


    remove_exif_data(os.path.join(src_folder, "Train"))
    remove_exif_data(os.path.join(src_folder, "Test"))


def remove_exif_data(src_folder):
    _,_, cat_images = next(os.walk(os.path.join(src_folder, "Cat")))
    for img in cat_images:
        try: piexif.remove(os.path.join(src_folder, "Cat", img ))

        except: pass

    
    _,_, dog_images = next(os.walk(os.path.join(src_folder, "Dog")))
    for img in dog_images:
        try: piexif.remove(os.path.join(src_folder, "Cat", img ))

        except: pass


