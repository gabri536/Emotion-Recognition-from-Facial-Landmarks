import os
import pandas as pd

def make_image_df(root_dir):
    """
    Walks through root_dir/<split>/<label> folders and returns
    a DataFrame with columns ['filepath', 'label', 'split'].
    """
    records = []
    # iterate over the two splits
    for split in ('train', 'test'):
        split_dir = os.path.join(root_dir, split)
        # iterate over each emotion subfolder
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            # iterate over images in that folder
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(label_dir, fname)
                    records.append({
                        'filepath': path,
                        'label': label,
                        'split': split
                    })
    # build DataFrame
    df = pd.DataFrame.from_records(records)
    return df

root = 'FER-2013'
df = make_image_df(root)

# split into train/test DataFrames
train_df = df[df['split'] == 'train'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)

print(f"Found {len(train_df)} training images, {len(test_df)} test images.")

# Optionally, save to disk for quick reloading later:
train_df.to_csv('fer2013_train.csv', index=False)
test_df.to_csv('fer2013_test.csv',  index=False)