# Pflow

## Install

```
pip install -e '.[dev]'
```

## Setup
```
cp .env.default .env
```

## Run
```
pflows doc/examples/birds-grouped-categories.json
```

In this example, we defined a workflow to download the dataset from Roboflow
as we can see in the image below:

![Birds Dataset](./doc/images/roboflow.png)

The workflow is defined in a JSON file, where we define the steps to be executed:

```
[
  {
    "task": "roboflow_tools.download_dataset",
    "target_dir": "{{BASE_FOLDER}}/datasets/downloaded/cub200_parts-50",
    "url": "https://universe.roboflow.com/explainableai-lavbv/cub200_parts/dataset/50"
  },
  {
    "task": "yolo_v8.load_dataset",
    "folder_path": "{{BASE_FOLDER}}/datasets/downloaded/cub200_parts-50"
  },
  {
    "task": "base.count_images"
  },
  {
    "task": "base.count_categories"
  },
  {
    "task": "categories.group_categories",
    "groups": {
      "upper": [["eye", "bill", "head", "nape", "throat"]],
      "lower": [["belly", "feet", "tail"]],
      "middle": [["Wing", "breast", "back"]],
      "Wing": [["Wing"]],
      "back": [["back"]],
      "belly": [["belly"]],
      "bill": [["bill"]],
      "eye": [["eye"]],
      "feet": [["feet"]],
      "head": [["head"]],
      "nape": [["nape"]],
      "tail": [["tail"]],
      "throat": [["throat"]]
    },
    "condition": "any"
  },
  {
    "task": "categories.keep",
    "categories": ["upper", "lower", "middle"]
  },
  {
    "task": "base.count_images"
  },
  {
    "task": "base.count_categories"
  },
  {
    "task": "base.show_categories"
  },
  {
    "task": "yolo_v8.write",
    "target_dir": "{{BASE_FOLDER}}/datasets/processed/birds-grouped-categories-cub200_parts-50"
  }
]
```

The workflow is composed of the following steps:

1. Download the dataset from Roboflow
2. Load the dataset
3. Count the number of images
4. Count the number of categories
5. Group the categories, to create new categories based on the existing ones (upper, lower, middle)
6. Keep only the categories that are in the groups "upper", "lower" and "middle"
7. Count the number of images
8. Count the number of categories
9. Show the categories
10. Write the dataset to disk

As we can see, we can use the `{{BASE_FOLDER}}` variable to refer to the base folder of the project.
These variables are defined in the `.env` file, which is used to configure the project.

We can see the input images and output images below:

#### Example Bird 1:

![Bird 1 before](./doc/images/bird1_before.png)
![Bird 1 before](./doc/images/bird1_after.png)

#### Example Bird 2:

![Bird 2 before](./doc/images/bird2_before.png)
![Bird 2 before](./doc/images/bird2_after.png)
