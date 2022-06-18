## HAM10000

Download and extract HAM10000 dataset here. Dataset can be downloaded from here:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Required files are:
  * HAM10000_images_part_1.zip
  * HAM10000_images_part_2.zip
  * HAM10000_metadata

Place those files here and execute `extract_dataset.sh` script.

Here is how everything should end up like:
```
- dataset
  - HAM10000
    - images
      - ISIC_0024306.jpg
      - ISIC_0024307.jpg
      - ...
    - HAM10000_metadata
```