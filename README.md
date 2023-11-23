# Indexify Extractors

## Overview

This repository hosts a collection of extractors for Indexify, a multi-modal structured extraction and real-time indexing engine designed for Large Language Model applications.
These extractors complement the core functionalities of Indexify, enabling seamless integration and enhanced data processing capabilities.

For the main Indexify project, visit: [Indexify Main Repository](https://github.com/diptanu/indexify).

## Features

- **Real-Time Data Processing:** Enhance Indexify with real-time data extraction capabilities.
- **Plug-and-Play Integration:** Easily integrate these extractors with the main Indexify engine.
- **Diverse Data Support:** Extends Indexify's ability to handle various data types and structures.

## Usage

Pick any extractor you are interested in running. You can also clone the entire repository to access all extractors.

```bash
git clone https://github.com/your-repository/indexify-extractors.git
```

From the Indexify Main Repository, you can then package the extractor (this creates a docker image), and then run it inside Indexify.
The easiest way to package and validate that it's running is

```bash
indexify extractor package --config-path indexify-extractors/extractors/minilm_l6.yaml
docker run diptanu/minilm-l6-extractor --text "Hello, I will be transformed into a vector-embedding!"
```

## Directory Structure

Currently there is a single folder `extractors/`. Each extractor inside it consists of a `.yaml` file, and a `.py` file.
The `.yaml` file describes the name, description, version, as well as the pip dependencies and the (ubuntu) system dependencies.
The containers are dockerized and are running Ubuntu::22.04 as of the day of writing.
