# Dataset

This folder contains the data for use in the project, i'm gonna use those datasets:

- **SIDD**: [Smartphone Image Denoising Dataset (SIDD)](https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset/data)
- **CBSD68**: [CBSD68 Dataset](https://github.com/clausmichele/CBSD68-dataset/tree/master)

The folder structure should be organized:

```bash
data/
│── SIDD_small/
│   ├── noisy/
│   ├── clean/
│
│── CBSD68/
│   ├── original/     
│   ├── noisy25/    
│
│── sort_data.py
```

## Sorting Data

After downloading the datasets, you will need to organize them. The sort_data.py script is provided for this purpose. Simply run the script, and it will automatically sort the datasets into the appropriate folders, for now the script only works for SIDD dataset. 