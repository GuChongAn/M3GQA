# M3GQA: A Multimodal Multi-hop and Knowledge Graph-based Framework for Question Answering
The official implementation of "M3GQA: A Multimodal Multi-hop and Knowledge Graph-based Framework for Question Answering"

### Project Structure

```shell
- /dataset
	- /manymodalqa (defalut ManymodalQA dataset)
- /src
	- /data
		manymodalQA.py
	- /model
		templateManager.py
	- /temp
	test.py
	utils.py
```

### Get Started

- Place the dataset in the above project structure, or write your own data loader in `src/dataset`
- Install the python libraries need to use in our code

- Execute test file `python test.py`.  If the test file is executed successfully, the answer to the question is stored in the file stored in `/temp`

### New Multimodal Dataset

#### Dataset

- In the `./HotpotQA` folder, we present an example of our dataset. 
- We have augmented the HotpotQA dataset by incorporating tabular and image data, linking them to document titles and each question-answer pair. 

#### construction code

- In the `./MMCraw` folder, we provide code demo for collecting image and table data from Wikipedia to extend the dataset for three text-only datasets such as 2WikiMultimodalQA（2WMQA）
