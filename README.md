#  MGHR: Multi-Granularity Hierarchical Reasoning for Visual Question Answering

# Abstract

Bridging the semantic gap between vision and language remains a fundamental challenge for Visual Question Answering (VQA) systems. Most existing methods rely on single-granularity cross-modal interactions to create unified cross-modality representation for reasoning answer. However, the reliance on the representations from single-granularity interactions may introduce redundant content to single object, which hinders fine-grained multi-hop reasoning across multiple objects and undermines the capture of global semantic information. To alleviate these issues, we propose a Multi-Granularity Hierarchical Reasoning framework (MGHR) to comprehensively achieve visual-linguistic collaboration inferring at three granularities: global, object, and relational, for the VQA task. In MGHR, we first propose a pre-trained visual transformer to extract global semantic information and generate high-level representations to infer the questions at global granularity. Meanwhile, we introduce an attention-based filtering mechanism to filter out question-irrelevant regions to eliminate redundant content for accurate alignment of object granularity. Then we construct a scene graph to concisely depict the spatial and relational information of objects in image, and develop a Graph Attention Network to propagate contextual information through adjacent nodes and edges in scene graph to implement multi-hop reasoning across objects at relational granularity. Finally, a confidence-weighted semantic fusion module is introduced to adaptively integrate inference results across the three granularities, yielding the final answer. We conduct extensive evaluations on the publicly available VQAv2, GQA, and VQA-CP benchmarks. Experimental results demonstrate that MGHR consistently outperforms state-of-the-art models. Ablation studies further validate the effectiveness of each module of MGHR.
<embed src="MGHRV2.drawio.pdf" type="application/pdf" width="100%" height="600px" />

## Requirements

```python
conda create -n VLM python==3.8

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install torch_geometric==2.4.0

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip3 install -r requirements.txt

pip install en_core_web_sm-3.7.1-py3-none-any.whl
```

### Dataset

- Download the raw images from the corresponding websites and place them in the `images` folder.

- Download the json files we provided

- Organize these files like this

  ```
  .cache/huggingface/transformers/openai/clip-vit-base-patch32/
  MGHR/
  	4m_base_finetune/
  		vqa/*.th
      data/
      	en_core_web_sm-3.7.1-py3-none-any.whl
          vqa/
              ...
          gqa/
          	...
          finetune/
          	*.json
      images/
          coco/
              train2014/*.jpg
              val2014/*.jpg
              test2015/*.jpg
          gqa/
              images/*.jpg
          ...
  
  ```

  Additionally, please note that some paths in the code need to be changed to the actual paths on your computer.

  ### Training

  To train the model using train set of VQA dataset, please follow:

  ```
  sh train.sh
  ```
  
  ### Testings
  
  To test the model using train set of VQA dataset, please follow:
  
  ```
  sh test.sh
```
  

