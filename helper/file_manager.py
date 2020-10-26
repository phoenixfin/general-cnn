import os
import random
import shutil


def create_data_dir(target_dir, subdirs):
    def trymakedir(sdir):
        try:
            os.mkdir(sdir)
        except:
            pass

    try:
        trymakedir(target_dir)
        subdir1 = ['train', 'validation']
        subdir2 = subdirs
        for sd1 in subdir1:
            trymakedir(target_dir+'/'+sd1)
            for sd2 in subdir2:
                subpath = target_dir+'/'+sd1+'/'+sd2
                trymakedir(subpath)
                for file in os.listdir(subpath):
                    os.remove(subpath+'/'+file)
    except OSError:
        pass


def split_data(source, train, validation, split_size):
    source_list = os.listdir(source)
    randomized = random.sample(source_list, len(source_list))
    filtered = [file for file in randomized if (
        os.path.getsize(source+file) != 0)]
    training_num = round(len(filtered)*split_size)
    for idx, img in enumerate(filtered):
        if idx < training_num:
            shutil.copyfile(source+img, train+img)
        else:
            shutil.copyfile(source+img, validation+img)


def arrange_data(base_dir, source, category, ratio=0.8):
    base = os.path.join(data_dir, name)
    base_dir = base+'\\tobeused'
    source_dir = base+'\\original'
    create_data_dir(base_dir, category)

    for cat in category:
        source_dir = source + '\\' + cat + '\\'
        train_dir = base_dir + '\\train\\' + cat + '\\'
        validation_dir = base_dir + '\\validation\\' + cat + '\\'
        split_data(source_dir, train_dir, validation_dir, split_size=ratio)


if __name__ == "__main__":
    data_dir = os.getcwd()+'\\data'

    # name = 'dogs-vs-cats'
    # category = ['cats','dogs']

    name = 'rps'
    category = ['rock', 'scissors', 'paper']

    arrange_data(name, category)
