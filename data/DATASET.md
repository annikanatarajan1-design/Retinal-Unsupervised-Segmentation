# Dataset

The OCT retinal images used in this project were obtained from a university laboratory dataset.

Due to usage restrictions, the full dataset cannot be publicly distributed in this repository.

## Dataset Details

- Number of images: 256
- Format: PNG
- Average image size: ~260 KB

```markdown

## Expected Folder Structure

Place the images inside:

data/images/

Example:

data/
 └── images/
      ├── NORMAL1.png
      ├── NORMAL2.png
      └── ...

## Using Your Own Dataset

Users can run the pipeline with their own OCT images by placing them in:

data/images/

The images will automatically be loaded by the dataset loader.
